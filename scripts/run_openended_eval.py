"""
scripts/run_openended_eval.py
-----------------------------
Open-ended evaluation entrypoint for AITR_Lab02.

Usage:
  python scripts/run_openended_eval.py --model mistral --rag false --n 500 --seed 42
  python scripts/run_openended_eval.py --model llama   --rag bm25  --n 500 --seed 42

Outputs:
  results/openended_results_{model}_{rag_tag}.csv
  results/checkpoint_oe_{model}_{rag_tag}_{n}.json (every 50 items)
"""

import argparse
import json
import os
import random
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data_loader import load_pubmedqa_test
from src.evaluate import token_f1, compute_openended_metrics
from src.prompt_templates import OPENENDED_TEMPLATE, format_context

MODEL_IDS = {
    "llama":   "meta-llama/Llama-3.1-8B-Instruct",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
}
FALLBACK_IDS = {
    "llama":   "BioMistral/BioMistral-7B",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
}


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_model(model_name: str):
    hf_id = MODEL_IDS[model_name]
    print(f"Loading model: {hf_id}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(hf_id)
        model = AutoModelForCausalLM.from_pretrained(
            hf_id, load_in_8bit=True, device_map="auto"
        )
    except Exception as e:
        print(f"WARNING: Failed to load {hf_id}: {e}")
        fallback = FALLBACK_IDS[model_name]
        print(f"Using fallback: {fallback}")
        tokenizer = AutoTokenizer.from_pretrained(fallback)
        model = AutoModelForCausalLM.from_pretrained(
            fallback, load_in_8bit=True, device_map="auto"
        )
    model.eval()
    return model, tokenizer


def generate(model, tokenizer, prompt: str, max_new_tokens: int = 100) -> str:
    inputs = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=1024
    ).to("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def categorize_failure(pred: str, gold: str, f1: float) -> str:
    """
    Categorize low-F1 open-ended outputs for deeper analysis.
    Returns one of: hallucination, incomplete, off_topic, near_miss, ok
    """
    if f1 >= 0.3:
        return "ok"
    pred_lower = pred.lower()
    gold_lower = gold.lower()
    # Near-miss: some overlap but low F1
    if f1 >= 0.1:
        return "near_miss"
    # Very short output
    if len(pred.split()) < 5:
        return "incomplete"
    # No keyword overlap at all
    gold_keywords = set(gold_lower.split()[:10])
    pred_keywords = set(pred_lower.split()[:20])
    if not gold_keywords & pred_keywords:
        return "off_topic"
    return "hallucination"


def main():
    parser = argparse.ArgumentParser(description="Open-ended evaluation on PubMedQA")
    parser.add_argument("--model",  required=True, choices=["llama", "mistral"])
    parser.add_argument("--rag",    required=True, choices=["false", "bm25"])
    parser.add_argument("--n",      type=int, default=500)
    parser.add_argument("--seed",   type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    use_rag = args.rag == "bm25"
    rag_tag = "rag" if use_rag else "norag"
    out_csv = f"results/openended_results_{args.model}_{rag_tag}.csv"
    os.makedirs("results", exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Open-Ended Evaluation | model={args.model} | rag={args.rag} | n={args.n} | seed={args.seed}")
    print(f"Output: {out_csv}")
    print(f"{'='*60}\n")

    items = load_pubmedqa_test(n=args.n)

    bm25, doc_texts = None, None
    if use_rag:
        from src.rag import build_bm25_index, retrieve
        bm25, doc_texts = build_bm25_index(n_test=args.n)

    model, tokenizer = load_model(args.model)

    preds, gold_answers, rows = [], [], []
    start_time = time.time()

    for i, item in enumerate(items):
        q       = item["question"]
        ctx     = format_context(item.get("context", {}), max_words=400)
        gold_of = item["long_answer"]

        if use_rag:
            retrieved = retrieve(bm25, doc_texts, q, top_k=1)
            ctx = (retrieved + " " + ctx)[:800]
        else:
            ctx = ctx[:800]

        prompt = OPENENDED_TEMPLATE.format(question=q, context=ctx)
        pred_of = generate(model, tokenizer, prompt, max_new_tokens=100)
        f1 = token_f1(pred_of, gold_of)
        failure_cat = categorize_failure(pred_of, gold_of, f1)

        preds.append(pred_of)
        gold_answers.append(gold_of)
        rows.append({
            "item_id":      i,
            "question":     q[:100],
            "gold_of":      gold_of[:200],
            "pred_of":      pred_of[:200],
            "token_f1":     f1,
            "failure_cat":  failure_cat,
        })

        if (i + 1) % 10 == 0:
            elapsed = time.time() - start_time
            mean_f1 = sum(r["token_f1"] for r in rows) / len(rows)
            print(f"  [{i+1}/{args.n}] Mean Token F1: {mean_f1:.4f} | Elapsed: {elapsed:.0f}s")

        if (i + 1) % 50 == 0:
            ckpt_path = f"results/checkpoint_oe_{args.model}_{rag_tag}_{i+1}.json"
            with open(ckpt_path, "w") as f:
                json.dump(rows, f)
            print(f"  Checkpoint saved: {ckpt_path}")

    metrics = compute_openended_metrics(preds, gold_answers)
    print(f"\n{'='*60}")
    print(f"FINAL RESULTS — {args.model.upper()} | RAG={use_rag}")
    print(f"  Mean Token F1:  {metrics['mean_token_f1']:.4f}")
    print(f"  Min Token F1:   {metrics['min_token_f1']:.4f}")
    print(f"  Max Token F1:   {metrics['max_token_f1']:.4f}")
    print(f"  N items:        {metrics['n_items']}")
    print(f"{'='*60}\n")

    # Deeper analysis: show 20 worst failure cases
    df = pd.DataFrame(rows)
    df["mc_model"]  = args.model
    df["rag"]       = args.rag
    df["mean_token_f1"] = metrics["mean_token_f1"]

    worst_20 = df.nsmallest(20, "token_f1")[["item_id","question","gold_of","pred_of","token_f1","failure_cat"]]
    failure_summary = df["failure_cat"].value_counts().to_dict()

    print("=== FAILURE CASE ANALYSIS (20 lowest F1 items) ===")
    print(worst_20.to_string(index=False))
    print(f"\nFailure category distribution: {failure_summary}")

    df.to_csv(out_csv, index=False)
    print(f"\nResults saved to: {out_csv}")

    summary = {
        "model": args.model,
        "rag": args.rag,
        "n": args.n,
        "seed": args.seed,
        **{k: v for k, v in metrics.items() if k != "scores"},
        "failure_distribution": failure_summary
    }
    summary_path = f"results/summary_oe_{args.model}_{rag_tag}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
