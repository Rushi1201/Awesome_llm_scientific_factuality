# Research Gaps and Future Directions

## 1. Dynamic Benchmarks

Scientific knowledge changes. Benchmarks should be periodically refreshed using newly published literature so that correct answers do not become stale.

## 2. Contamination Resistance

A benchmark can lose diagnostic value if questions or near-duplicates appear in training data. Future datasets should track provenance and use contamination-resistant construction.

## 3. Claim-Level Evidence

Response-level correctness is too coarse for long answers. Evaluations should decompose outputs into claims and identify the evidence supporting or contradicting each claim.

## 4. Calibration and Abstention

A model should receive credit for declining to answer when its evidence is insufficient. Evaluation should therefore distinguish incorrect confidence from appropriate uncertainty.

## 5. Retrieval vs Generation

RAG systems have at least two failure points: retrieving poor evidence and generating claims that do not faithfully follow good evidence. Benchmarks should score both components separately.

## 6. Multimodal Scientific Factuality

Scientific information is often expressed through figures, tables, chemical structures, microscopy images, and plots. Text-only factuality benchmarks do not fully evaluate these settings.

## 7. Expert Human Evaluation

Automated judges are useful for scale but are themselves imperfect. Meta-evaluation against domain experts remains necessary, especially for high-stakes scientific and medical claims.

## 8. Cross-Domain Generalization

A benchmark should test whether a factuality method transfers across biology, medicine, physics, chemistry, and other scientific fields rather than overfitting to one corpus.
