"""
src/data_loader.py
------------------
PubMedQA loading utilities for AITR_Lab02_RushikeshShirsath_LLMFactuality.

VERIFIED: PubMedQA pqa_labeled split has 1,000 expert-labeled items.
Test partition: last 500 items (index-based, fixed, deterministic).
Train partition: first 500 items (used for BM25 corpus only — never test items).
"""

from datasets import load_dataset


def load_pubmedqa_test(n: int = 500) -> list:
    """
    Load the last n items from PubMedQA pqa_labeled split as the fixed test set.

    Args:
        n: Number of test items (default 500 per configs/experiment_config.yaml)

    Returns:
        List of dicts with keys: pubid, question, context, long_answer, final_decision
    """
    ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
    items = list(ds)
    test_items = items[-n:]

    # Sanity checks
    assert len(test_items) == n, f"Expected {n} test items, got {len(test_items)}"
    assert "long_answer" in test_items[0], "Missing 'long_answer' field"
    assert test_items[0]["final_decision"] in ["yes", "no", "maybe"], \
        "Unexpected label value"

    return test_items


def load_pubmedqa_train(n: int = 500) -> list:
    """
    Load the first (total - n) items as the training/retrieval corpus.
    These items are NEVER used as test items — no leakage.

    Args:
        n: Number of items reserved for test (default 500)

    Returns:
        List of dicts for use as BM25 retrieval corpus
    """
    ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
    items = list(ds)
    train_items = items[:-n]

    assert len(train_items) == len(items) - n
    return train_items


def print_dataset_summary():
    """Print a summary of the dataset for verification."""
    ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
    items = list(ds)
    labels = [item["final_decision"] for item in items]
    yes_count = labels.count("yes")
    no_count = labels.count("no")
    maybe_count = labels.count("maybe")
    majority_acc = yes_count / len(items)

    print(f"Total items:        {len(items)}")
    print(f"Yes:                {yes_count} ({yes_count/len(items)*100:.1f}%)")
    print(f"No:                 {no_count} ({no_count/len(items)*100:.1f}%)")
    print(f"Maybe:              {maybe_count} ({maybe_count/len(items)*100:.1f}%)")
    print(f"Majority-class acc: {majority_acc*100:.1f}%  (expected ~55.0% — VERIFY)")
    print(f"Keys:               {list(items[0].keys())}")


if __name__ == "__main__":
    print_dataset_summary()

    test = load_pubmedqa_test(n=500)
    train = load_pubmedqa_train(n=500)
    print(f"\nTest items:  {len(test)}")
    print(f"Train items: {len(train)}")
    print("data_loader.py: ALL CHECKS PASSED")
