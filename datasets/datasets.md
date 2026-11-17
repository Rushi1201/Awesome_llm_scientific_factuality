# Datasets

## 1. SciFact
- **Purpose:** Scientific claim verification.
- **Content:** Expert-written scientific claims paired with evidence-containing abstracts and support/refute labels.
- **Source:** Allen Institute for AI.
- **Link:** https://github.com/allenai/scifact
- **Why useful:** Directly evaluates literature-grounded scientific factuality.

## 2. PubMedQA
- **Purpose:** Biomedical research question answering.
- **Content:** Questions derived from PubMed abstracts with yes/no/maybe answers and evidence.
- **Source:** PubMedQA project.
- **Link:** https://github.com/pubmedqa/pubmedqa
- **Why useful:** Tests factual reasoning over biomedical research evidence.

## 3. GPQA
- **Purpose:** Graduate-level scientific reasoning.
- **Content:** 448 expert-authored multiple-choice questions in biology, physics, and chemistry.
- **Source:** GPQA authors.
- **Link:** https://github.com/idavidrein/gpqa
- **Why useful:** Designed to be difficult even for knowledgeable non-experts with web access.

## 4. TruthfulQA
- **Purpose:** General truthfulness.
- **Content:** Questions designed around common misconceptions.
- **Link:** https://github.com/sylinrl/TruthfulQA
- **Why useful:** Provides a broad factuality baseline.

## 5. SimpleQA
- **Purpose:** Short-form factuality.
- **Content:** Fact-seeking questions with correct, incorrect, or not-attempted grading.
- **Link:** https://github.com/openai/simple-evals
- **Why useful:** Tests whether a model knows specific facts and can avoid unsupported attempts.

## 6. RAGTruth
- **Purpose:** Hallucination evaluation in retrieval-augmented generation.
- **Content:** Nearly 18,000 naturally generated RAG responses with word-level hallucination annotations.
- **Link:** https://github.com/ParticleMedia/RAGTruth
- **Why useful:** Separates RAG grounding from generic response quality.

## 7. FreshQA / FreshLLMs
- **Purpose:** Dynamic factuality.
- **Content:** Questions involving changing world knowledge and false-premise correction.
- **Link:** https://github.com/freshllms/freshqa
- **Why useful:** Addresses the staleness problem in static benchmarks.

## 8. SciFact-Open
- **Purpose:** Open-domain scientific claim verification.
- **Content:** Evaluation over roughly 500K research abstracts.
- **Link:** https://github.com/dwadden/scifact-open
- **Why useful:** Tests whether systems generalize from small claim-verification datasets to large scientific corpora.
