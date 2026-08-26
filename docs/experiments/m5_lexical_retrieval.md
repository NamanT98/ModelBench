# M5 — Lexical Few-Shot Retrieval

## Problem Investigated

With the M4-B.1 deterministic schema-linking strategy successfully pushing SQL validity to 83.5%, the majority of remaining ModelBench evaluation failures stemmed from semantic misunderstandings (incorrect JOIN logic, missed aggregations, or mishandled filtering). The goal of M5 was to determine if adding retrieved, relevant SQL demonstrations from a training corpus could improve the LLM's logical reasoning and execution accuracy.

**Hypothesis:** Relevant demonstrations can improve Text-to-SQL generation, particularly for JOINs, aggregations, HAVING, nested queries, and other reasoning-heavy SQL patterns.

## Architecture

M5 introduces a dynamic few-shot retrieval component:
- **Retrieval Method**: A deterministic `JaccardSimilarityRetriever`. It tokenizes target questions using our existing `nltk` pipeline and computes the Jaccard similarity against pre-tokenized questions in the `train_spider.json` dataset.
- **Leakage Prevention**: The evaluation evaluation (`dev.json`) is strictly separated from the retrieval index (`train_spider.json`). Strict validation tests enforce that no dev samples can ever enter the retriever's training index.
- **Performance Caching**: To circumvent the heavy computational cost of running the NLTK tokenization pipeline over 7,000 training samples at the start of every run, the index is pre-computed and serialized locally.
- **Prompt Format**: The dynamic few-shot prompt constructs `EXAMPLE` blocks for the top `k` retrieved samples. Each block consists of the linked schema (generated specifically for that example), the natural language question, and the gold SQL query.

## Experimental Setup

M4-B.1 settings remained strictly unchanged:
- **Model**: `Qwen/Qwen2.5-Coder-3B-Instruct`
- **Schema Strategy**: `schema_linking_normalized` (M4-B.1)
- **Dataset**: Full 1,034-sample Spider dev split

## Results

| Metric | M4-B.1 (Baseline) | M5 (k=1) | M5 (k=3) |
| :--- | :--- | :--- | :--- |
| **Execution Accuracy** | 42.5% | 53.8% (+11.3%) | **55.13% (+12.6%)** |
| **SQL Validity** | 83.5% | 85.1% (+1.6%) | 84.04% (+0.5%) |
| **Exact Match** | 12.1% | 22.9% (+10.8%) | **27.85% (+15.8%)** |
| **Avg Generation Latency** | ~1.20s | 1.40s | 1.52s |
| **Avg Retrieval Latency** | - | 0.065s | 0.065s |

## Key Findings

The hypothesis is strongly validated. Providing relevant Text-to-SQL demonstrations yields a massive **12.6% absolute increase in execution accuracy**. Furthermore, the exact match rate more than doubled (from 12.1% to 27.85%), demonstrating that seeing proper SQL dialects and structural examples drastically aligned the 3B model's output syntax with the expected ground-truth queries.

## Limitations

While the Jaccard similarity retriever is lightning-fast (~65ms) and deterministic, it is inherently limited by exact token matching. It cannot retrieve examples based on semantic similarity (e.g., retrieving an example about "cars" when asked about "vehicles") or structural complexity.
