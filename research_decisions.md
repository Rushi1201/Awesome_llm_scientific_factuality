# Research Decisions Log
# AITR_Lab02_RushikeshShirsath_LLMFactuality
# Student: Rushikesh Shirsath | M.Tech (IT) Robotics & AI | IIIT Allahabad

---

## Decision 1 — Topic Continuity

| Field | Detail |
|---|---|
| **Decision** | Remain in LLM factuality benchmarking domain from MRA_01 |
| **Reason** | Deep literature familiarity; MRA_01 provides verified source evidence for every gap without requiring a fresh literature search |
| **Source / Evidence** | MRA_01 §5 explicitly lists the MC ↔ long-form accuracy gap as a structural limitation of current benchmarks |
| **Evidence Label** | VERIFIED |
| **Next Action** | Narrow to one measurable, computationally testable manifestation of this gap |

---

## Decision 2 — Gap Selection

| Field | Detail |
|---|---|
| **Decision** | Focus on the multiple-choice (MC) accuracy vs. open-ended token F1 discrepancy, with and without RAG |
| **Reason** | (a) VERIFIED in original paper [7] — Singhal et al. 2023 report Flan-PaLM's strong MC performance alongside clinician-rated long-form quality gaps. (b) Computationally testable with public datasets. (c) Directly tests a claim from MRA_01 without requiring novel architecture. |
| **Source / Evidence** | MRA_01 §5: "strong performance on multiple-choice factual recall tasks does not guarantee factually reliable long-form generation... Singhal et al. (2023) found meaningful gaps between Flan-PaLM's strong MultiMedQA multiple-choice performance and clinician assessments of its open-ended answers." Citation traceable to [7]. |
| **Evidence Label** | VERIFIED (from original source [7]); INFERRED that it generalizes to open-weight model families |
| **Next Action** | Confirm PubMedQA has long-answer fields usable for open-ended F1 evaluation |

---

## Decision 3 — Pathway Choice (B: Controlled Comparative Study)

| Field | Detail |
|---|---|
| **Decision** | Use Pathway B (controlled comparative study) rather than Pathway A (reproduction + extension) or Pathway C (targeted modification) |
| **Reason** | Pathway A requires exact replication of a closed-weight proprietary model (Flan-PaLM), which is not feasible. Pathway C risks confounding variables from architectural changes. Pathway B with open-weight models is most feasible and directly answers RQ1–RQ3. |
| **Source / Evidence** | Assignment §3, Pathway B: "Evaluate relevant methods under a common, carefully chosen protocol that answers a specific question." |
| **Evidence Label** | INFERRED (methodology choice; no VERIFIED claim needed) |
| **Next Action** | Confirm Llama-3.1-8B-Instruct and Mistral-7B-Instruct-v0.3 are accessible on HuggingFace |

---

## Decision 4 — Dataset Selection (PubMedQA)

| Field | Detail |
|---|---|
| **Decision** | Use PubMedQA `pqa_labeled` split (yes/no/maybe; with long-answer gold field) |
| **Reason** | (a) Directly referenced in MRA_01 §3.1 — "Jin et al. (2019) established an early paradigm in which models must answer yes/no/maybe research questions using the corresponding PubMed abstract as context." (b) Has BOTH MC labels AND long free-text gold answers, uniquely enabling the same test items to serve both evaluation formats. (c) Publicly available, CC-BY 4.0 license. (d) Well-known majority-class baseline of 55.0% provides a verifiable lower bound. |
| **Source / Evidence** | VERIFIED: accessed at https://huggingface.co/datasets/qiaojin/PubMedQA; long_answer field confirmed populated on all 1,000 expert-labeled items. |
| **Evidence Label** | VERIFIED |
| **Next Action** | Confirm the `long_answer` field yields non-trivial token F1 scores on a 10-item manual check |

---

## Decision 5 — Model Selection

| Field | Detail |
|---|---|
| **Decision** | Use Llama-3.1-8B-Instruct and Mistral-7B-Instruct-v0.3 as the two open-weight model families |
| **Reason** | Both are (a) open-weight, (b) instruction-tuned, (c) runnable on Colab T4 with 8-bit quantization, (d) well-known in the community for fair cross-model comparison. |
| **Source / Evidence** | VERIFIED: Mistral-7B-Instruct-v0.3 — Apache 2.0, no gating. Llama-3.1-8B-Instruct — Llama 3 Community License, requires HF token. |
| **Evidence Label** | VERIFIED (access confirmed on HuggingFace) |
| **Fallback** | If Llama-3.1-8B requires HF Pro tier, substitute BioMistral-7B (open, no token gate, biomedical fine-tuned) |
| **Next Action** | Load both models in 8-bit mode and confirm they fit in T4 VRAM |

---

## Decision 6 — RAG Implementation (BM25 over Training Abstracts)

| Field | Detail |
|---|---|
| **Decision** | Use BM25 (rank_bm25 library) over PubMedQA training abstracts as the RAG component |
| **Reason** | (a) BM25 is a standard, interpretable sparse retriever — appropriate as a lower-bound RAG baseline. (b) Dense retrieval (FAISS + embeddings) would exceed Colab free-tier memory. (c) Retrieval corpus is restricted to training partition only to prevent leakage. |
| **Source / Evidence** | INFERRED that BM25 represents a valid lower bound; no specific paper tested BM25 on PubMedQA for this protocol |
| **Evidence Label** | INFERRED |
| **Limitation** | BM25 may underestimate RAG benefit — explicitly flagged in paper limitations |
| **Next Action** | Build BM25 index over training abstracts; manually verify 10 retrievals for quality |

---

## Decision 7 — Test Partition (Last 500 Items)

| Field | Detail |
|---|---|
| **Decision** | Use the last 500 items of the pqa_labeled split as the fixed test partition |
| **Reason** | Compute budget (Colab T4, ~2 hrs per model-condition). Fixed by index order (deterministic, no random sampling needed). Leaves remaining items for BM25 retrieval corpus. |
| **Source / Evidence** | INFERRED from compute constraints |
| **Evidence Label** | INFERRED |
| **Risk** | 500-item sample may not fully represent full 1,000-item distribution — mitigated by reporting bootstrap 95% CI |
| **Next Action** | Freeze partition before any results are viewed |

---

## Decision 8 — Evaluation Metrics

| Field | Detail |
|---|---|
| **Decision** | MC: 3-class accuracy + macro-F1 (yes/no/maybe). Open-ended: token-level F1 (SQuAD-style) against gold long_answer field |
| **Reason** | (a) MC 3-class accuracy is the standard PubMedQA metric (matches [3]). (b) Token F1 is a standard open-ended QA metric that does not require human evaluation, making it reproducible. |
| **Source / Evidence** | Token F1 implementation VERIFIED against SQuAD scorer on 10 manual examples (±0.02 agreement) |
| **Evidence Label** | VERIFIED (implementation); INFERRED (suitability of token F1 for PubMedQA long_answer) |
| **Limitation** | Token F1 is a shallow metric and may not capture semantic correctness — BERTScore recommended for future work |
| **Next Action** | Run token F1 on 10 manual examples to confirm sensible score range (expected 0.1–0.5) |
