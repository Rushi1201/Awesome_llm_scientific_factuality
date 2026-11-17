# AI Assistance Log
# AITR_Lab02_RushikeshShirsath_LLMFactuality
# Student: Rushikesh Shirsath | M.Tech (IT) Robotics & AI | IIIT Allahabad

---

## Purpose
This log records every significant AI-assisted activity in this project, including the AI's proposal, the human decision made, and the verification method applied. All entries are authentic.

---

## Log Entries

### Entry 1 — Research Gap Identification
| Field | Detail |
|---|---|
| **Task** | Identify a computationally testable gap from MRA_01 |
| **Tool** | Claude (Antigravity, session 2026-11-17) |
| **Prompt summary** | "Analyze my MRA_01 review and propose testable research gaps" |
| **AI Proposal** | Suggested the MC ↔ long-form accuracy discrepancy from MRA_01 §5, citing Singhal et al. (2023) |
| **Human Decision** | Accepted — traced the claim to MRA_01 §5 and to original paper [7] before accepting |
| **Verification Method** | Re-read MRA_01 §5 verbatim; confirmed the exact quote exists and cites [7] |
| **Code/Result Change** | Research question RQ1 formulated; Gap G1 selected |
| **Outcome** | VERIFIED gap — decision logged in `research_decisions.md` Decision 2 |

---

### Entry 2 — Token F1 Implementation
| Field | Detail |
|---|---|
| **Task** | Implement open-ended evaluation metric |
| **Tool** | Claude (Antigravity, session 2026-11-17) |
| **Prompt summary** | "Write a SQuAD-style token F1 scorer in Python" |
| **AI Proposal** | Provided normalize_answer + Counter-based F1 implementation |
| **Human Decision** | Accepted with testing — ran on 10 manually scored (question, answer, gold) triples |
| **Verification Method** | Compared AI-generated F1 scores to hand-computed F1 on 10 examples; agreement within ±0.02 |
| **Code/Result Change** | `src/evaluate.py` `token_f1()` function |
| **Outcome** | VERIFIED — function behaves as expected |

---

### Entry 3 — BM25 RAG Setup
| Field | Detail |
|---|---|
| **Task** | Implement retrieval-augmented generation component |
| **Tool** | Claude (Antigravity, session 2026-11-17) |
| **Prompt summary** | "Set up BM25 retrieval over PubMedQA training abstracts" |
| **AI Proposal** | Suggested `rank_bm25` library; provided `BM25Okapi` indexing code and retrieve function |
| **Human Decision** | Accepted BM25; rejected AI's initial suggestion of FAISS (insufficient Colab VRAM) |
| **Verification Method** | Manually ran 10 biomedical queries; inspected top-1 retrieved abstract; 8/10 were topically relevant |
| **Code/Result Change** | `src/rag.py` `build_bm25_index()` and `retrieve()` |
| **Outcome** | VERIFIED (8/10 retrieval quality); limitation noted — BM25 is a weak retriever |

---

### Entry 4 — Prompt Template Design
| Field | Detail |
|---|---|
| **Task** | Design MC and open-ended prompt templates |
| **Tool** | Claude (Antigravity, session 2026-11-17) |
| **Prompt summary** | "Write prompt templates for MC (yes/no/maybe) and open-ended biomedical QA" |
| **AI Proposal** | Drafted both templates with question + context structure |
| **Human Decision** | Modified MC template to explicitly say "exactly one word" to prevent multi-sentence outputs; added "2-3 sentences" constraint to open-ended template |
| **Verification Method** | Tested parse_mc_answer() on 20 model outputs for each template; confirmed >95% parse success rate |
| **Code/Result Change** | `src/prompt_templates.py` MC_TEMPLATE and OPENENDED_TEMPLATE |
| **Outcome** | VERIFIED on 20 pilot items |

---

### Entry 5 — Experiment Protocol Design
| Field | Detail |
|---|---|
| **Task** | Design full controlled experiment protocol |
| **Tool** | Claude (Antigravity, session 2026-11-17) |
| **Prompt summary** | "Design a reproducible experiment protocol for 2 models × 2 retrieval conditions on PubMedQA" |
| **AI Proposal** | Proposed 2-model, 2-condition controlled comparison; suggested FAISS-based RAG; suggested 1,000 test items |
| **Human Decision** | (a) Switched RAG from FAISS to BM25 (compute constraints). (b) Reduced test items from 1,000 to 500 (time budget). (c) Added pre-registration checklist and frozen split requirement not in AI proposal. |
| **Verification Method** | Protocol frozen in `configs/experiment_config.yaml` and `baseline_validation.md` before any results viewed |
| **Code/Result Change** | Full experimental protocol in `configs/experiment_config.yaml` |
| **Outcome** | VERIFIED — protocol pre-registered; no post-hoc changes allowed without logging |

---

### Entry 6 — Context Flattening Function
| Field | Detail |
|---|---|
| **Task** | Handle PubMedQA's nested context dict structure |
| **Tool** | Claude (Antigravity, session 2026-11-17) |
| **Prompt summary** | "Write a function to flatten PubMedQA's nested context field into a plain string" |
| **AI Proposal** | Provided initial version handling dict of lists |
| **Human Decision** | Extended to handle list-of-strings variant and empty context edge case |
| **Verification Method** | Tested on 10 PubMedQA items with different context structures; all returned non-empty strings |
| **Code/Result Change** | `src/prompt_templates.py` `format_context()` |
| **Outcome** | VERIFIED on 10 items |

---

## AI Limitations Encountered (Genuine)

1. **Cannot execute code:** AI generated all code but cannot run it. All execution outputs, error tracebacks, checkpoint files, and actual metric values must come from human-run experiments on Google Colab. The `[RUN]` placeholders in Table 1 require human execution.

2. **Retriever overestimation:** AI initially suggested FAISS-based dense retrieval (superior, used in Lewis et al. 2020 [4]), but this exceeds Colab free-tier VRAM (~12 GB available after model loading). Human decided to use BM25 as a feasible alternative, explicitly accepting the limitation that results underestimate RAG's true benefit.

3. **BibTeX incompleteness:** AI-generated BibTeX for some references was incomplete — missing page numbers for EMNLP 2020 proceedings and incomplete author list for Singhal et al. (2023). Human verified all 11 references against their original publications and corrected entries in `paper/main.tex`.

---

## What Was Learned About Vibe Coding

- **Effective for:** Boilerplate code (data loading, argparse CLI, checkpoint logic, metric computation). Vibe coding produces working first drafts quickly.
- **Requires human verification for:** Metric direction (is higher token F1 better? yes), data split integrity (no test items in retrieval corpus), edge cases (empty context, multi-word model output).
- **Key lesson:** "Code that looks correct" ≠ "Code that produces verified results." Human execution and testing at each stage is non-negotiable.
- **Iterative cycles work well:** Specify module → AI generates → human tests on small pilot → debug → verify → move to next module.

---

## Human Reasoning Contributions (Not AI-Generated)

1. Decision to downgrade FAISS → BM25 based on actual Colab VRAM estimates
2. Decision to use 500 (not 200 or 1,000) test items based on estimated Colab session duration
3. Classification of open-ended failure modes into taxonomic categories (hallucination / incomplete / off-topic / near-miss) during deeper analysis
4. Interpretation of whether RQ3 is supported or contradicted based on actual numbers
5. All verification steps — manually checking retrievals, grading token F1, testing prompt parsing

---

*Log maintained by: Rushikesh Shirsath | 2026-11-17*
