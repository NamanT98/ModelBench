# Analysis: Lexical vs. Semantic Retrieval Overlap (M5 ∪ M6)

## Executive Summary

To understand *why* the overall accuracy difference between Lexical Retrieval (M5) and Semantic Retrieval (M6) was small (+0.8%), we performed a sample-by-sample overlap analysis of the correctly executed queries across the 1,034-sample Spider dev set.

The analysis revealed massive divergence. Lexical and Semantic retrieval strategies successfully resolve **different subsets of failures**. 

## Overlap Breakdown

*   **Total Correct (M5 ∪ M6)**: 657
*   **Both Correct (M5 ∩ M6)**: 491
*   **Only M5 Correct**: 79 (Lexical won)
*   **Only M6 Correct**: 87 (Semantic won)

The divergence is significant. 79 queries were uniquely correctly answered by Jaccard retrieval, while 87 were uniquely correctly answered by Embedding retrieval. 

If we had an "oracle" retriever that perfectly picked between the two strategies for every query, the execution accuracy would jump to **63.5%** (657/1034), significantly higher than M6's independent 55.90%.

## Analysis: Why do they diverge?

### 1. Lexical Strengths (M5)
Jaccard similarity excels when the user's natural language question uses exact table names, column names, or specific data values that are rare in the corpus. It effectively retrieves demonstrations that use those exact schemas, reinforcing structural mapping. M6 (semantic) often fails here because it considers exact schema names less semantically "weighty" than the logical phrasing of the sentence.

### 2. Semantic Strengths (M6)
Embeddings excel when the user asks a complex abstract question. For example, "Who is the oldest?" logically maps to `ORDER BY age DESC LIMIT 1`. Embeddings map the conceptual intent of the query to a demonstration with the same logical structure, even if the schema vocabulary (e.g., "oldest employee" vs "longest serving president") is entirely different. Lexical retrieval completely fails on these queries because there is zero vocabulary overlap.

## Conclusion

Relying exclusively on semantic embeddings discards valuable exact-match lexical signals, while relying purely on lexical matching fails on abstract logic. The two strategies are highly complementary. 

Future experiments must investigate **Hybrid Retrieval** strategies that combine both signals to rank few-shot demonstrations, aiming to approach the theoretical 63.5% accuracy ceiling.
