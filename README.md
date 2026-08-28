# Awesome Factuality Benchmarking of Large Language Models

A curated research repository for benchmarking factual accuracy of large language models (LLMs) across scientific domains. It connects an AI-assisted research paper and citation-integrity audit with a reusable collection of verified papers, datasets, tools, implementations, and learning resources.

> **Topic:** Benchmarking Factual Accuracy of Large Language Models Across Scientific Domains  
> **Student:** Rushikesh Pandharinath Shirshath  
> **Roll Number:** MRA2026001  
> **Programme:** IT Robotics and AI  
> **Topic ID:** 1

## Contents
- [Overview](#overview)
- [AI-Assisted Research Paper](#ai-assisted-research-paper)
- [Citation Integrity Audit](#citation-integrity-audit)
- [Survey Papers](#survey-papers)
- [Foundational Papers](#foundational-papers)
- [Recent Research Papers](#recent-research-papers)
- [Datasets](#datasets)
- [Tools and Libraries](#tools-and-libraries)
- [GitHub Implementations](#github-implementations)
- [Tutorials and Learning Resources](#tutorials-and-learning-resources)
- [License](#license)

## Overview
Large language models can produce fluent and useful scientific answers while still making unsupported or incorrect factual claims. Factuality benchmarking therefore asks a different question from ordinary language-quality evaluation: **does the generated statement agree with a reliable ground truth or identifiable evidence?** This becomes especially difficult in scientific domains because claims often require specialist knowledge, careful interpretation of literature, and evidence that can change as new studies are published.

The literature has developed several complementary benchmark families. General-purpose benchmarks such as TruthfulQA and SimpleQA test whether models avoid false or unsupported answers. Scientific and biomedical benchmarks such as SciFact, PubMedQA, MultiMedQA, and GPQA introduce domain knowledge and expert-level questions. Other resources evaluate long-form factuality, citation quality, hallucination detection, or evidence attribution.

A central lesson across this repository is that **benchmark score alone is not equivalent to scientific reliability**. Multiple-choice accuracy can hide unsupported reasoning in long-form answers; static benchmarks can become contaminated or stale; automated evaluators can themselves make mistakes; and evidence retrieval can fail even when the final answer sounds plausible. Robust evaluation therefore benefits from dynamic datasets, expert annotation, claim-level evidence, uncertainty-aware scoring, retrieval evaluation, and human verification.

## AI-Assisted Research Paper
**Title:** Benchmarking Factual Accuracy of Large Language Models Across Scientific Domains  
**Abstract/Description:** [Insert a 2-4 sentence short abstract/description of what your AI-generated paper discusses regarding factuality benchmarking.]  
[View AI-Assisted Paper](paper/AI_Assisted_Research_Paper.pdf)

## Citation Integrity Audit
This audit systematically checked 10 AI-generated references to verify their authenticity and claim support. All 10 sampled references were classified as **A (VERIFIED)**, resulting in an authenticity score of **100/100** and a prediction accuracy of **100%**.  
[View Citation Integrity Audit](citation-audit/Citation_Integrity_Audit.pdf)

## Survey Papers
*Note to student: Add your survey papers here using the exact required format below.*

**[Paper Title]**  
[Authors], [Year], [Journal/Conference]  
[Paper / DOI](Link)  
*One-line explanation of why the paper is relevant.*

## Foundational Papers
*Note to student: Add your foundational methodology papers here using the same format.*

**[Paper Title]**  
[Authors], [Year], [Journal/Conference]  
[Paper / DOI](Link)  
*One-line explanation of why the paper is relevant.*

## Recent Research Papers
*Note to student: Add your recent research and application papers here using the same format. Ensure your total paper count across these three sections equals at least 20 verified scholarly papers.*

**[Paper Title]**  
[Authors], [Year], [Journal/Conference]  
[Paper / DOI](Link)  
*One-line explanation of why the paper is relevant.*

## Datasets
*Note to student: You must include the Name, Source, Description, Application, and Link for each dataset. Below are your selected datasets properly formatted as a template.*

*   **SciFact** - [Source/Link]: [Brief description of the dataset and its application/use.]
*   **PubMedQA** - [Source/Link]: [Brief description of the dataset and its application/use.]
*   **GPQA** - [Source/Link]: [Brief description of the dataset and its application/use.]
*   **TruthfulQA** - [Source/Link]: [Brief description of the dataset and its application/use.]
*   **SimpleQA** - [Source/Link]: [Brief description of the dataset and its application/use.]
*   **RAGTruth** - [Source/Link]: [Brief description of the dataset and its application/use.]
*   **FreshQA / FreshLLMs** - [Source/Link]: [Brief description of the dataset and its application/use.]
*   **SciFact-Open** - [Source/Link]: [Brief description of the dataset and its application/use.]

## Tools and Libraries
*Note to student: You must include the Purpose and Official/Project Link for each tool.*

*   **Hugging Face Transformers** - [Link]: [Brief description of its purpose.]
*   **Hugging Face Evaluate** - [Link]: [Brief description of its purpose.]
*   **Sentence Transformers** - [Link]: [Brief description of its purpose.]
*   **FAISS** - [Link]: [Brief description of its purpose.]
*   **Pyserini** - [Link]: [Brief description of its purpose.]
*   **Ragas** - [Link]: [Brief description of its purpose.]
*   **LlamaIndex** - [Link]: [Brief description of its purpose.]

## GitHub Implementations
*Note to student: You must include what it implements and why it is relevant, along with the link.*

*   **AllenAI SciFact** - [Link]: [What it implements and why it is relevant.]
*   **TruthfulQA** - [Link]: [What it implements and why it is relevant.]
*   **PubMedQA** - [Link]: [What it implements and why it is relevant.]
*   **GPQA** - [Link]: [What it implements and why it is relevant.]
*   **RARR** - [Link]: [What it implements and why it is relevant.]
*   **QAFactEval** - [Link]: [What it implements and why it is relevant.]
*   **RAGTruth** - [Link]: [What it implements and why it is relevant.]

## Tutorials and Learning Resources
*Note to student: You must include an authoritative resource link and a brief purpose.*

*   **Hugging Face NLP course** - [Link]: [Brief purpose.]
*   **RAG documentation and guides** - [Link]: [Brief purpose.]
*   **Sentence Transformers documentation** - [Link]: [Brief purpose.]
*   **Haystack tutorials** - [Link]: [Brief purpose.]
*   **LlamaIndex documentation** - [Link]: [Brief purpose.]
*   **LangChain retrieval tutorials** - [Link]: [Brief purpose.]

## License
This repository's original written material is released under **CC BY 4.0**. Third-party papers, datasets, software, and documentation remain under their respective licenses. This repository links to external resources rather than redistributing third-party copyrighted papers.
