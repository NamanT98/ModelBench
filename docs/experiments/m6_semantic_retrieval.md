# M6 — Semantic Embedding Retrieval

## Problem Investigated

Lexical retrieval (M5) improved Text-to-SQL generation significantly (+12.6% execution accuracy), but it relies strictly on exact token matching. The M6 experiment evaluated whether dense semantic embeddings could provide higher-quality few-shot demonstrations by matching the underlying *intent* and *meaning* of the user's natural language question, rather than just vocabulary overlap.

**Hypothesis**: Semantic retrieval will outperform lexical retrieval by mapping abstract reasoning patterns (e.g., "oldest" mapping to `ORDER BY age DESC LIMIT 1`) even when vocabulary diverges completely.

## Experimental Setup

M6 directly compared `BAAI/bge-small-en-v1.5` dense embeddings against the M5 Jaccard baseline.
- **Model**: `Qwen/Qwen2.5-Coder-3B-Instruct`
- **Schema Strategy**: `schema_linking_normalized` (M4-B.1)
- **Dataset**: Full 1,034-sample Spider dev split
- **Retrieval Metric**: Cosine Similarity (via dot-product over normalized embeddings)
- **k (demonstrations)**: 3

## Results

| Metric | M4-B.1 (Zero-Shot) | M5 (k=3 Jaccard) | M6 (k=3 Embedding) |
| :--- | :--- | :--- | :--- |
| **Execution Accuracy** | 42.5% | 55.13% (570/1034) | **55.90%** (578/1034) |
| **Exact Match** | 12.1% | **27.85%** | 27.47% |
| **SQL Validity** | 83.5% | **85.30%** | 84.04% |
| **Average Latency** | - | 64.63 ms | 57.27 ms |

## Key Findings

1. **Modest Absolute Improvement**: Semantic Embeddings (M6) achieved a slight overall accuracy gain (+0.8%) over Lexical Jaccard (M5). 
2. **Syntax Regression**: M6 saw a slight regression in exact SQL string matches and syntax validity. This suggests that while embeddings help the model reason to the correct answer logically, lexical similarity provides better exact syntactical and table-name templates for the specific domain.
3. **High Performance**: The embedding approach was slightly faster than M5 (57ms vs 65ms) because vector dot-products using `numpy` are highly optimized, whereas M5 relies on `nltk` Python loops.

## Complementarity (The Overlap Insight)

The most critical finding from M6 is the **high degree of complementarity** between the two approaches. Over 15% of all successful predictions were uniquely solved by only one of the two strategies, indicating that lexical and semantic signals capture fundamentally different aspects of query alignment.

*(See `analysis/m6_retrieval_overlap.md` for the detailed overlap analysis)*.
