# Benchmarking Factual Accuracy of Large Language Models Across Scientific Domains

## Abstract

Large language models (LLMs) are increasingly used for scientific information seeking, but fluent generation does not guarantee factual correctness. This review examines benchmarks and methods for evaluating factual accuracy across scientific and general knowledge domains. It covers TruthfulQA, SimpleQA, GPQA, PubMedQA, MultiMedQA, SciFact, SciFact-Open, FELM, FActScore, ALCE, RAGTruth, and related hallucination-evaluation approaches. The literature suggests that factuality is multidimensional: short-form correctness, long-form claim support, citation quality, retrieval faithfulness, and expert scientific reasoning measure related but distinct capabilities. Current benchmarks also face contamination, static ground truth, expert-annotation cost, and limitations of automated evaluators. Future evaluation should combine dynamic evidence, claim-level attribution, uncertainty-aware scoring, multimodal scientific content, and human expert oversight.

## 1. Introduction

LLMs can produce answers that are coherent and persuasive even when individual claims are unsupported. In scientific settings, this is particularly important because incorrect statements may be difficult for non-experts to detect and because scientific claims often require specialized evidence. Factuality benchmarking therefore aims to measure whether a model's output agrees with reliable knowledge or evidence rather than merely whether it is linguistically well formed.

## 2. General-Purpose Factuality Benchmarks

TruthfulQA tests whether models avoid common misconceptions, while SimpleQA focuses on short fact-seeking questions with relatively unambiguous answers. These benchmarks are useful for broad comparisons but do not fully capture the evidence-grounded reasoning required by scientific literature.

## 3. Scientific and Biomedical Evaluation

SciFact frames factuality as scientific claim verification against research abstracts. SciFact-Open extends this setting to a much larger open-domain scientific corpus. PubMedQA tests biomedical research question answering, while MultiMedQA combines medical examination and consumer-health settings with expert-oriented evaluation. GPQA targets difficult graduate-level questions in biology, physics, and chemistry.

## 4. Fine-Grained and Long-Form Factuality

FActScore evaluates long-form outputs by decomposing them into atomic claims. FELM provides fine-grained factuality annotations and reference links. These approaches are important because a long answer may contain both correct and incorrect statements, making a single response-level label insufficient.

## 5. Retrieval and Citation Grounding

Retrieval-augmented generation can provide evidence for model outputs, but retrieval quality and generation faithfulness are separate failure points. ALCE evaluates answers together with citations, while RAGTruth provides hallucination annotations in retrieval-augmented settings. RARR and Chain-of-Verification illustrate post-generation strategies for attribution and verification.

## 6. Research Challenges

Important limitations include benchmark contamination, static ground truth, expensive expert annotation, and imperfect automated factuality judges. Scientific knowledge changes over time, so a benchmark that was correct when constructed may become incomplete or outdated. Multiple-choice accuracy may also overestimate real-world reliability because it does not necessarily measure evidence attribution or long-form reasoning.

## 7. Future Directions

Promising directions include dynamic benchmarks refreshed from current literature, evidence-linked claim-level annotations, uncertainty and abstention metrics, multimodal scientific evaluation, and standardized meta-evaluation of automated factuality judges. Retrieval precision, evidence faithfulness, and final-answer correctness should be measured separately.

## Conclusion

Benchmarking LLM factual accuracy requires more than one benchmark or one score. A robust evaluation ecosystem should combine general factuality tests, scientific expert questions, literature-grounded claim verification, long-form claim analysis, retrieval evaluation, citation quality, and human expert assessment.
