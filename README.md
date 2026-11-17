# Evaluating Factual Accuracy in Biomedical LLMs: A Comparative Study of Multiple-Choice and Open-Ended Questions

**Student:** Rushikesh Shirsath | M.Tech (IT) Robotics & AI | IIIT Allahabad  
**Course:** AI Tools for Research (Post Mid-Sem Assignment — MRA_02)  
**Submission Tag:** AITR_Lab02_RushikeshShirsath_LLMFactuality

---

## Research Question

> To what extent does a model's accuracy on a multiple-choice biomedical QA benchmark (PubMedQA) predict its factual reliability on a matched open-ended generation task, and does retrieval-augmented generation (RAG) narrow or widen this gap?

**Sub-questions:**
- RQ2: Does model family (Llama vs. Mistral) affect the size of the MC ↔ open-ended discrepancy?
- RQ3: Does BM25 RAG improve open-ended token F1 more than it improves MC accuracy?

**Study pathway:** B — Controlled Comparative Study  
**Gap source:** MRA_01 §5, traceable to Singhal et al. (2023) [7]

---

## Repository Structure

```
AITR_Lab02_RushikeshShirsath_LLMFactuality/
├── README.md                   ← this file
├── requirements.txt            ← verified dependencies
├── research_decisions.md       ← literature-to-experiment decision log
├── baseline_validation.md      ← reproduction checks and sanity tests
├── ai_assistance_log.md        ← significant prompts and human verification
├── configs/
│   └── experiment_config.yaml  ← frozen experimental settings
├── data/
│   └── README.md               ← PubMedQA access instructions (no raw data committed)
├── notebooks/
│   ├── 01_environment_test.ipynb
│   ├── 02_baseline_mc.ipynb
│   ├── 03_openended_eval.ipynb
│   └── 04_rag_pipeline.ipynb
├── src/
│   ├── data_loader.py          ← PubMedQA loading utilities
│   ├── prompt_templates.py     ← MC and open-ended prompt templates
│   ├── evaluate.py             ← MC accuracy + token F1 evaluators
│   └── rag.py                  ← BM25 retriever
├── scripts/
│   ├── run_mc_eval.py          ← MC evaluation entrypoint
│   ├── run_openended_eval.py   ← open-ended evaluation entrypoint
│   └── plot_results.py         ← generate figures from results CSV
├── results/
│   ├── mc_results.csv          ← [generated after run]
│   ├── openended_results.csv   ← [generated after run]
│   └── figures/                ← [generated after run]
└── paper/
    └── main.tex                ← LaTeX manuscript
```

---

## Setup

### Requirements
- Python 3.10+
- CUDA-capable GPU with ≥ 16 GB VRAM (Google Colab T4 recommended)

### Install Dependencies
```bash
pip install -r requirements.txt
```

### HuggingFace Token (for Llama-3.1-8B — gated model)
```bash
export HF_TOKEN=your_huggingface_token_here
# Or in Colab:
# from huggingface_hub import login; login(token="your_token")
```
> **Fallback:** If Llama-3.1-8B is unavailable, substitute `BioMistral/BioMistral-7B` (open, no token required).

---

## Dataset

PubMedQA is loaded automatically via HuggingFace Datasets — **no manual download needed**.

```python
from datasets import load_dataset
ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled")
```

- **License:** CC-BY 4.0
- **Source:** https://pubmedqa.github.io/
- **No restricted raw data is committed to this repository.**
- Test partition: last 500 items of `pqa_labeled` split (fixed by index, seed=42)

---

## Run Experiments

### Step 0: Environment Test
```bash
python notebooks/01_environment_test.ipynb
# OR
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```
Expected: `True  Tesla T4`

### Step 1: MC Baseline (no RAG)
```bash
python scripts/run_mc_eval.py --model mistral --rag false --n 500 --seed 42
python scripts/run_mc_eval.py --model llama --rag false --n 500 --seed 42
```

### Step 2: Open-Ended Baseline (no RAG)
```bash
python scripts/run_openended_eval.py --model mistral --rag false --n 500 --seed 42
python scripts/run_openended_eval.py --model llama --rag false --n 500 --seed 42
```

### Step 3: RAG Experiments [TEST REQUIRED — not yet executed on full set]
```bash
python scripts/run_mc_eval.py --model mistral --rag bm25 --n 500 --seed 42
python scripts/run_mc_eval.py --model llama --rag bm25 --n 500 --seed 42
python scripts/run_openended_eval.py --model mistral --rag bm25 --n 500 --seed 42
python scripts/run_openended_eval.py --model llama --rag bm25 --n 500 --seed 42
```

### Step 4: Generate Figures
```bash
python scripts/plot_results.py --input results/all_results.csv --output results/figures/
```

---

## Minimum Reproduction Checklist (for a second student)

- [ ] Python 3.10+ installed
- [ ] CUDA-capable GPU available (≥ 16 GB VRAM)
- [ ] `pip install -r requirements.txt` succeeds
- [ ] HuggingFace token with Llama-3.1-8B access obtained (or use Mistral-7B as fallback)
- [ ] Run environment test — confirm CUDA True and T4 GPU
- [ ] Run pilot (n=20): `python scripts/run_mc_eval.py --model mistral --rag false --n 20 --seed 42`
- [ ] Confirm MC accuracy > 55% and token F1 > 0.1 on 20-item pilot
- [ ] Run full 500-item evaluation for all 4 conditions
- [ ] Verify `results/mc_results.csv` and `results/openended_results.csv` are populated

---

## Compute Requirements

| Component | Requirement |
|---|---|
| GPU | NVIDIA T4 (16 GB) or better |
| Inference mode | 8-bit quantized (bitsandbytes) — no fine-tuning |
| Estimated time | ~2 hours per model-condition pair |
| Total estimated | ~8–10 hours for all 4 conditions |
| Colab session | Free-tier T4; use checkpointing (every 50 items) |

---

## Limitations

1. **BM25 is a weak retriever** — results may underestimate RAG benefit vs. dense retrieval (FAISS)
2. **Token F1 is a shallow metric** — does not capture semantic correctness; BERTScore recommended as future work
3. **500-item sample** — may not fully represent the full 1,000-item distribution; bootstrap CIs reported
4. **8-bit quantization** — may affect generation quality vs. full-precision inference
5. **No comparison to closed-weight models** — GPT-4 comparison is out of scope for this compute budget

---

## Credits & Licenses

| Resource | Source | License |
|---|---|---|
| PubMedQA dataset | Jin et al. (2019) | CC-BY 4.0 |
| Llama-3.1-8B-Instruct | Meta AI | Llama 3 Community License |
| Mistral-7B-Instruct-v0.3 | Mistral AI | Apache 2.0 |
| HuggingFace Transformers | HuggingFace | Apache 2.0 |
| rank_bm25 | dorianbrown | Apache 2.0 |

---

## Citation

```
Shirsath, R. (2026). MRA_02: Multiple-Choice vs. Open-Ended LLM Factuality on PubMedQA
— A Controlled Study with Retrieval Augmentation. IIIT Allahabad.
```

---

*No API keys, private data, restricted model weights, or secrets are stored in this repository.*
