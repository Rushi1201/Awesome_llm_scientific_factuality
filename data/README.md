# Data Directory — PubMedQA Access Instructions

## Dataset: PubMedQA

**No raw data files are stored in this repository.**  
The dataset is loaded automatically via HuggingFace Datasets.

---

## Loading Instructions

```python
from datasets import load_dataset

# Load the expert-labeled split (1,000 items)
ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")

# Inspect schema
print(ds[0].keys())
# Expected: ['pubid', 'question', 'context', 'long_answer', 'final_decision']

# Test partition used in this project: last 500 items
test_items  = list(ds)[-500:]
train_items = list(ds)[:-500:]   # used for BM25 retrieval corpus only
```

---

## Dataset Details

| Field | Value |
|---|---|
| **Name** | PubMedQA |
| **HuggingFace path** | `qiaojin/PubMedQA` |
| **Config used** | `pqa_labeled` |
| **Total items** | 1,000 expert-labeled |
| **License** | CC-BY 4.0 |
| **Official site** | https://pubmedqa.github.io/ |
| **Paper** | Jin et al. (2019). PubMedQA: A Dataset for Biomedical Research Question Answering. EMNLP. |

---

## Field Descriptions

| Field | Type | Description |
|---|---|---|
| `pubid` | int | PubMed article ID |
| `question` | str | Biomedical yes/no/maybe question |
| `context` | dict | Nested dict containing PubMed abstract sentences |
| `long_answer` | str | Gold free-text answer (used for open-ended token F1) |
| `final_decision` | str | Gold MC label: "yes", "no", or "maybe" |

---

## Context Field Note

The `context` field is a nested dictionary. Use `src/prompt_templates.format_context()` to flatten it:

```python
from src.prompt_templates import format_context
ctx_str = format_context(item['context'], max_words=400)
```

---

## Majority-Class Baseline

Predicting "yes" for all items yields ~55.0% accuracy — matches the value reported in Jin et al. (2019).  
This is the lower bound for all MC experiments in this project.

---

## Ethical Notes

- PubMedQA contains biomedical research question-answer pairs derived from PubMed abstracts.
- No patient data, personally identifiable information, or restricted clinical records are included.
- License (CC-BY 4.0) permits research use with attribution.
- No data is modified, re-distributed, or committed to this repository.
