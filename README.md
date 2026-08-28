# Awesome Factuality Benchmarking of Large Language Models

A curated research repository for **benchmarking factual accuracy of large language models (LLMs) across scientific domains**. It connects the earlier AI-assisted research paper and citation-integrity audit with a reusable collection of verified papers, datasets, tools, implementations, and learning resources.

> **Topic:** Benchmarking Factual Accuracy of Large Language Models Across Scientific Domains  
> **Student:** Rushikesh Pandharinath Shirshath  
> **Roll Number:** MRA2026001  
> **Programme:** IT Robotics and AI  
> **Topic ID:** 1

## Contents

- [Topic Overview](#topic-overview)
- [AI-Assisted Research Paper](#ai-assisted-research-paper)
- [Citation Integrity Audit](#citation-integrity-audit)
- [Curated Research Papers](#curated-research-papers)
- [Datasets](#datasets)
- [Tools and Libraries](#tools-and-libraries)
- [GitHub Implementations](#github-implementations)
- [Tutorials and Learning Resources](#tutorials-and-learning-resources)
- [Research Gaps](#research-gaps)
- [Repository Structure](#repository-structure)
- [License](#license)

## Topic Overview

Large language models can produce fluent and useful scientific answers while still making unsupported or incorrect factual claims. Factuality benchmarking therefore asks a different question from ordinary language-quality evaluation: **does the generated statement agree with a reliable ground truth or identifiable evidence?** This becomes especially difficult in scientific domains because claims often require specialist knowledge, careful interpretation of literature, and evidence that can change as new studies are published.

The literature has developed several complementary benchmark families. General-purpose benchmarks such as TruthfulQA and SimpleQA test whether models avoid false or unsupported answers. Scientific and biomedical benchmarks such as SciFact, PubMedQA, MultiMedQA, and GPQA introduce domain knowledge and expert-level questions. Other resources evaluate long-form factuality, citation quality, hallucination detection, or evidence attribution, including FActScore, ALCE, FELM, SciFact-Open, RAGTruth, and SelfCheckGPT.

A central lesson across this repository is that **benchmark score alone is not equivalent to scientific reliability**. Multiple-choice accuracy can hide unsupported reasoning in long-form answers; static benchmarks can become contaminated or stale; automated evaluators can themselves make mistakes; and evidence retrieval can fail even when the final answer sounds plausible. Robust evaluation therefore benefits from dynamic datasets, expert annotation, claim-level evidence, uncertainty-aware scoring, retrieval evaluation, and human verification.

This repository organizes these resources so that a researcher can move from a benchmark paper to its dataset, implementation, evaluation method, and practical learning material.

## AI-Assisted Research Paper

**Benchmarking Factual Accuracy of Large Language Models Across Scientific Domains**  
*AI-assisted review prepared during Lab 1.*

The original paper is included in `paper/AI_Assisted_Research_Paper.md`. The repository preserves the original reference inventory rather than silently replacing it.

## Citation Integrity Audit

The earlier lab contained an 11-reference AI-generated paper and a systematic audit of 10 references. The provided worksheet recorded all ten sampled references as classification **A (VERIFIED)** and reported an authenticity score of **100/100** and prediction accuracy of **100%**.

The original audit evidence is summarized in:

- `citation-audit/Citation_Integrity_Audit.md`

The audit should be treated as the student's prior experimental record. This repository's curated bibliography is separately organized and includes persistent identifiers or authoritative links where available.

## Curated Research Papers

The repository contains **20 research papers**, grouped by purpose:

1. Surveys and foundations
2. General factuality benchmarks
3. Scientific and biomedical benchmarks
4. Claim verification and citation evaluation
5. Hallucination detection and mitigation

See `references/references.md`.

## Datasets

Minimum required: 3. Included:

- SciFact
- PubMedQA
- GPQA
- TruthfulQA
- SimpleQA
- RAGTruth
- FreshQA / FreshLLMs
- SciFact-Open

See `datasets/datasets.md`.

## Tools and Libraries

Minimum required: 5. Included:

- Hugging Face Transformers
- Hugging Face Evaluate
- Sentence Transformers
- FAISS
- Pyserini
- Ragas
- LlamaIndex

See `tools/tools.md`.

## GitHub Implementations

Minimum required: 5. Included implementations include:

- AllenAI SciFact
- TruthfulQA
- PubMedQA
- GPQA
- RARR
- QAFactEval
- RAGTruth

See `implementations/github-repositories.md`.

## Tutorials and Learning Resources

Minimum required: 5. Included:

- Hugging Face NLP course
- RAG documentation and guides
- Sentence Transformers documentation
- Haystack tutorials
- LlamaIndex documentation
- LangChain retrieval tutorials

See `tutorials/learning-resources.md`.

## Research Gaps

The most important open directions identified from the literature are:

- dynamic and continuously refreshed scientific benchmarks;
- contamination-resistant evaluation;
- claim-level evidence attribution;
- calibration and abstention-aware scoring;
- multimodal scientific factuality;
- cross-domain expert evaluation;
- separate measurement of retrieval quality and generation faithfulness;
- rigorous validation of automated factuality judges.

A fuller discussion is in `docs/research-gaps.md`.

## Repository Structure

```text
awesome-factuality-llm/
├── README.md
├── LICENSE
├── paper/
│   └── AI_Assisted_Research_Paper.md
├── citation-audit/
│   └── Citation_Integrity_Audit.md
├── references/
│   └── references.md
├── datasets/
│   └── datasets.md
├── tools/
│   └── tools.md
├── implementations/
│   └── github-repositories.md
├── tutorials/
│   └── learning-resources.md
└── docs/
    ├── research-gaps.md
    └── commit-plan.md
```

## License

The repository's original written material is released under **CC BY 4.0**. Third-party papers, datasets, software, and documentation remain under their respective licenses. This repository links to external resources rather than redistributing third-party copyrighted papers.

## Submission

After creating the public GitHub repository, submit:

1. the public repository URL;
2. the GitHub profile URL.

Replace the profile/repository placeholders in `docs/submission.md` before submission.
