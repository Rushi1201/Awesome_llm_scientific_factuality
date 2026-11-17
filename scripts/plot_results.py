"""
scripts/plot_results.py
-----------------------
Generate publication-quality figures from experiment results.
AITR_Lab02_RushikeshShirsath_LLMFactuality.

Usage:
  python scripts/plot_results.py --input results/all_results.csv --output results/figures/

Figures generated:
  - figure1_bar_mc_vs_oe.png       : Bar chart MC accuracy vs. token F1 by model and RAG
  - figure2_confusion_matrices.png : Per-class MC confusion matrices
  - figure3_f1_distributions.png   : Token F1 score distributions (histogram)
"""

import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family":    "sans-serif",
    "font.size":      11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "figure.dpi":     150,
})
COLORS = {
    "llama_norag":   "#2196F3",
    "llama_rag":     "#0D47A1",
    "mistral_norag": "#FF9800",
    "mistral_rag":   "#E65100",
    "baseline":      "#9E9E9E",
}
LABELS = {
    "llama_norag":   "Llama-3.1-8B\n(no RAG)",
    "llama_rag":     "Llama-3.1-8B\n(BM25 RAG)",
    "mistral_norag": "Mistral-7B\n(no RAG)",
    "mistral_rag":   "Mistral-7B\n(BM25 RAG)",
    "baseline":      "Majority\nbaseline",
}


def load_summaries(results_dir: str) -> pd.DataFrame:
    """Load all summary JSON files from results directory."""
    rows = []
    for fname in os.listdir(results_dir):
        if fname.startswith("summary_") and fname.endswith(".json"):
            with open(os.path.join(results_dir, fname)) as f:
                rows.append(json.load(f))
    if not rows:
        raise FileNotFoundError(
            f"No summary_*.json files found in {results_dir}. "
            "Run the evaluation scripts first."
        )
    return pd.DataFrame(rows)


def figure1_bar_chart(df: pd.DataFrame, out_dir: str):
    """
    Figure 1: Grouped bar chart of MC accuracy and token F1 per condition.
    Shows the MC ↔ open-ended gap visually.
    """
    conditions = []
    mc_accs = []
    oe_f1s = []

    # Majority class baseline
    conditions.append("baseline")
    mc_accs.append(0.55)
    oe_f1s.append(0.00)

    for _, row in df.sort_values(["model", "rag"]).iterrows():
        key = f"{row['model']}_{row['rag'].replace('false','norag').replace('bm25','rag')}"
        conditions.append(key)
        mc_accs.append(row.get("accuracy", 0))
        oe_f1s.append(row.get("mean_token_f1", 0))

    x = np.arange(len(conditions))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    bars_mc = ax.bar(x - width/2, mc_accs, width, label="MC Accuracy (3-class)",
                     color=[COLORS.get(c, "#607D8B") for c in conditions], alpha=0.9)
    bars_oe = ax.bar(x + width/2, oe_f1s, width, label="Open-ended Token F1",
                     color=[COLORS.get(c, "#607D8B") for c in conditions], alpha=0.55,
                     hatch="///")

    ax.axhline(y=0.55, color="gray", linestyle="--", linewidth=1.2,
               label="Majority-class MC baseline (55%)")

    ax.set_xticks(x)
    ax.set_xticklabels([LABELS.get(c, c) for c in conditions], fontsize=9)
    ax.set_ylabel("Score")
    ax.set_title(
        "Figure 1: MC Accuracy vs. Open-Ended Token F1 by Model and RAG Condition\n"
        "(PubMedQA, n=500, seed=42)"
    )
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)

    # Annotate MC–OE gap on each experimental bar
    for i, (mc, oe) in enumerate(zip(mc_accs, oe_f1s)):
        if i == 0:
            continue
        gap = mc - oe
        ax.annotate(f"gap={gap:.2f}", xy=(x[i], max(mc, oe) + 0.02),
                    ha="center", fontsize=7.5, color="#333333")

    plt.tight_layout()
    out_path = os.path.join(out_dir, "figure1_bar_mc_vs_oe.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def figure2_confusion_matrices(results_dir: str, out_dir: str):
    """
    Figure 2: Per-class confusion matrices for MC evaluation.
    Loads mc_results_*.csv files.
    """
    mc_files = [f for f in os.listdir(results_dir)
                if f.startswith("mc_results_") and f.endswith(".csv")]
    if not mc_files:
        print("WARNING: No mc_results_*.csv files found — skipping Figure 2")
        return

    n_plots = len(mc_files)
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 4))
    if n_plots == 1:
        axes = [axes]

    for ax, fname in zip(axes, sorted(mc_files)):
        df = pd.read_csv(os.path.join(results_dir, fname))
        labels = ["yes", "no", "maybe"]
        cm = confusion_matrix(df["gold_mc"], df["pred_mc"], labels=labels)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=labels, yticklabels=labels, ax=ax)
        condition = fname.replace("mc_results_", "").replace(".csv", "")
        ax.set_title(f"{condition}")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Gold")

    fig.suptitle(
        "Figure 2: Per-Class MC Confusion Matrices by Condition\n"
        "(PubMedQA, n=500, seed=42)",
        y=1.02
    )
    plt.tight_layout()
    out_path = os.path.join(out_dir, "figure2_confusion_matrices.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def figure3_f1_distributions(results_dir: str, out_dir: str):
    """
    Figure 3: Token F1 score distributions (histograms) per condition.
    """
    oe_files = [f for f in os.listdir(results_dir)
                if f.startswith("openended_results_") and f.endswith(".csv")]
    if not oe_files:
        print("WARNING: No openended_results_*.csv files found — skipping Figure 3")
        return

    fig, axes = plt.subplots(1, len(oe_files), figsize=(5 * len(oe_files), 4), sharey=True)
    if len(oe_files) == 1:
        axes = [axes]

    for ax, fname in zip(axes, sorted(oe_files)):
        df = pd.read_csv(os.path.join(results_dir, fname))
        condition = fname.replace("openended_results_", "").replace(".csv", "")
        color = COLORS.get(condition, "#607D8B")
        ax.hist(df["token_f1"], bins=20, range=(0, 1), color=color, alpha=0.75, edgecolor="white")
        mean_f1 = df["token_f1"].mean()
        ax.axvline(mean_f1, color="red", linestyle="--", linewidth=1.5, label=f"Mean={mean_f1:.3f}")
        ax.set_title(condition)
        ax.set_xlabel("Token F1")
        ax.legend(fontsize=8)

    axes[0].set_ylabel("Count")
    fig.suptitle(
        "Figure 3: Token F1 Score Distributions by Condition\n"
        "(PubMedQA Open-Ended Evaluation, n=500, seed=42)",
        y=1.02
    )
    plt.tight_layout()
    out_path = os.path.join(out_dir, "figure3_f1_distributions.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate result figures")
    parser.add_argument("--input",  default="results/", help="Results directory")
    parser.add_argument("--output", default="results/figures/", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print("Loading summaries...")
    try:
        df = load_summaries(args.input)
        figure1_bar_chart(df, args.output)
    except FileNotFoundError as e:
        print(f"WARNING: {e}")
        print("Generating placeholder Figure 1 with dummy data...")
        # Generate a placeholder figure with [RUN] annotations
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.text(0.5, 0.5, "Run experiments first to generate real figures.\n"
                           "Execute scripts/run_mc_eval.py and run_openended_eval.py",
                ha="center", va="center", fontsize=14, transform=ax.transAxes,
                bbox=dict(boxstyle="round", facecolor="#fff3cd", alpha=0.8))
        ax.set_title("Figure 1: MC Accuracy vs. Open-Ended Token F1 [PENDING REAL DATA]")
        plt.savefig(os.path.join(args.output, "figure1_bar_mc_vs_oe.png"),
                    dpi=150, bbox_inches="tight")
        plt.close()

    figure2_confusion_matrices(args.input, args.output)
    figure3_f1_distributions(args.input, args.output)

    print(f"\nAll figures saved to: {args.output}")


if __name__ == "__main__":
    main()
