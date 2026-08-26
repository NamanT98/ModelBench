# Engineering Optimization: M7-RRF Candidate Pool

## The Problem

The initial full-ranking implementation of `M7 Hybrid RRF` introduced a massive O(N) Python overhead. For every query, it fetched, scored, and ranked all ~7,000 training examples from *both* the lexical and semantic retrievers before applying the RRF formula over the full 14,000-candidate union. This resulted in significant CPU bottlenecking during retrieval (averaging ~14 seconds per query).

## The Bounded Candidate-Pool Approach

To eliminate this overhead without degrading accuracy, we implemented a bounded candidate-pool (`candidate_n`). The logic operates as follows:
1. Fetch only the top-`N` candidates from the lexical and semantic retrievers.
2. Form the union (at most `2N` candidates).
3. Compute the standard RRF formula (`c=60`). If a candidate is missing from one retriever's top-N list, it contributes `0` to that half of the RRF score (we do not artificially inflate ranks).

## Optimization Results

We re-evaluated the full 1,034-sample Spider dev set across multiple bounded depths, comparing them against the full-ranking baseline.

| Candidate N   | Exec Acc (%) | SQL Valid (%) | Exact Match (%) | Retrieval Latency (s) | Exact Top-3 Match % |
|:--------------|-----------:|-------------:|--------------:|----------------------:|-------------------:|
| **25**        |      58.03 |        85.88 |         29.50 |                 ~0.24 |             100.00 |
| **50**        |      58.03 |        85.88 |         29.50 |                 ~0.24 |             100.00 |
| **100**       |      58.03 |        85.88 |         29.50 |                 ~0.24 |             100.00 |
| **200**       |      58.03 |        85.88 |         29.50 |                 ~0.24 |             100.00 |
| **Full (∞)**  |      58.03 |        85.88 |         29.50 |                 ~14.0 |             100.00 |

## Conclusion

The results are definitive: `candidate_n=25` was empirically equivalent to exhaustive RRF on the evaluated Spider dev set. Bounding the candidate pool to just 25 items from each retriever yielded a **100% exact identity match** for the final top-3 retrieved items compared to exhaustive full-corpus ranking. 

Because the RRF constant is 60, candidates ranked beyond the top 25 contribute so little to the final score that they can safely be ignored without altering the final top-K retrieved set.

By only considering `2N = 50` candidates instead of `14,000`, the retrieval latency is capped by the sheer speed of the SentenceTransformer encoding (~0.24s per query on CPU).

*(Note: This equivalence was demonstrated specifically for the Spider dataset size and a retrieval requirement of k=3. It does not generalize universally to massive corpora.)*
