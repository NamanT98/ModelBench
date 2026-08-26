# M7 — Hybrid Retrieval Strategies for Text-to-SQL

## Problem Investigated

Lexical retrieval (M5) and Semantic retrieval (M6) achieved similar overall execution accuracy (~55%), but an overlap analysis revealed they solved fundamentally different subsets of queries. M7 investigates whether combining the lexical matching strengths of Jaccard Similarity (exact schema vocabulary) with the semantic matching strengths of dense embeddings (abstract intent) yields a superior set of few-shot demonstrations for the `Qwen2.5-Coder-3B-Instruct` model.

**Hypothesis**: A hybrid retrieval strategy will outperform either individual strategy by combining orthogonal signals.

## Experimental Setup

- **Dataset**: Spider 1.0 (dev split, 1034 samples)
- **Model**: `Qwen/Qwen2.5-Coder-3B-Instruct`
- **Schema Strategy**: `schema_linking_normalized` (M4-B.1)
- **Generations**: `max_new_tokens=256`, `temperature=0.0`
- **k (demonstrations)**: 3

We evaluated two families of Hybrid Retrieval against the established M5 and M6 baselines:
1. **Hybrid Score Fusion**: Normalizing the Jaccard (lexical) and Cosine (semantic) scores, and fusing them via a weighted alpha: `(α * Lexical) + ((1 - α) * Semantic)`.
2. **Hybrid Reciprocal Rank Fusion (RRF)**: Discarding the raw scores entirely and ranking candidates based on their fused reciprocal rank: `1 / (k + rank)`.

## Benchmark Results

| Strategy                     |   SQL Validity (%) |   Exact Match (%) |   Execution Accuracy (%) |
|:-----------------------------|-------------------:|------------------:|-------------------------:|
| **M5 Jaccard (K=3)**            |              84.04 |             27.85 |                    55.13 |
| **M6 Embed (K=3)**           |              84.04 |             27.47 |                    55.90 |
| **M7 Hybrid Score (α=0.25)** |              84.82 |             29.01 |                    57.16 |
| **M7 Hybrid Score (α=0.50)** |              85.30 |             28.05 |                    55.03 |
| **M7 Hybrid Score (α=0.75)** |              85.01 |             28.05 |                    55.51 |
| **M7 Hybrid RRF**            |          **85.88** |         **29.50** |                **58.03** |

## Key Findings

### 1. Reciprocal Rank Fusion (RRF) is the Clear Winner
The `M7 Hybrid RRF` strategy outperformed all other methods across every single metric. It achieved a staggering **58.03% Execution Accuracy**, a solid ~2.5% absolute improvement over both isolated baselines. It also achieved the highest **SQL Validity (85.88%)** and highest **Exact Match (29.50%)**.

The important methodological finding is that RRF avoids directly combining incomparable lexical and semantic score distributions. Because Jaccard and Cosine similarities have fundamentally different distributions and scales, simple min-max normalization (`hybrid_score`) often fails to balance them effectively. RRF bypasses this by only comparing relative ranks.

### 2. Score-Based Fusion is Highly Sensitive to Alpha
For the `hybrid_score` strategy, weighting more heavily towards Semantic matching (`α=0.25`, where Lexical=0.25 and Semantic=0.75) performed significantly better (57.16%) than weighting towards Lexical matching (α=0.75 yielded 55.51%). This suggests that while lexical signals are important for tie-breaking and finding exact column matches, the primary relevance driver should be semantic similarity.

## Conclusion

Hybrid retrieval using Reciprocal Rank Fusion completely validates the hypothesis. By leveraging both exact-match vocabulary and dense intent representations, it produces a demonstrably superior few-shot prompt for Text-to-SQL generation.
