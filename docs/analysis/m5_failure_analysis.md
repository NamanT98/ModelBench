# Analysis: M4 Schema Linking Failure Modes

## Executive Summary

The M4 deterministic lexical schema linker was validated on the full 1,034-sample Spider dev set. While it successfully reduced the average schema context by 62.5% (from 4.4 tables to 2.2), execution accuracy remained at 32.1% (prior to NLP normalization). 

A detailed failure analysis reveals that **schema retrieval is not the primary bottleneck for this model once basic morphological normalization is applied**. Of the 702 execution failures observed in naive linking, 118 (11.4% of total) were caused by the linker failing to find any relevant tables. The vast majority of failures (88.6%) occurred when the linker provided a schema, but the model either hallucinated column names (Invalid SQL) or failed to reason correctly about JOINs and aggregations (Semantic SQL).

Notably, the linking failures were almost entirely due to trivial morphological variations (singular/plural) or direct synonyms, which were subsequently solved by M4-B.1 NLP normalization.

## Linking Failure Analysis (Pre-Normalization)

When the naive linker returned an empty schema (`linking_success=False`) for 118 samples (11.4%), manual analysis of a random sample revealed the root causes:

1. **Plural/Singular Mismatch (approx. 60%)**
   - *Example:* Question asks for "countries", schema has table `country`.
   - *Example:* Question asks for "songs", schema has table `song`.
2. **Direct Synonym Mismatch (approx. 30%)**
   - *Example:* Question asks for "students", schema has table `Highschooler`.
3. **Concept Mismatch (approx. 10%)**
   - *Example:* Question asks for "How many people live in...", gold SQL uses `SUM(Population)` from `country`.

This proved that a simple lemmatizer (e.g., stripping trailing 's') and a small synonym dictionary would eliminate almost all linking failures without the latency overhead of embeddings.

## Schema Selection Errors (Noisy/Incomplete)

When the linker found a match, it sometimes under-selected or over-selected:
- **Under-selection:** Question asks for "names of poker players", linker matches `poker_player` (finding the ID), but misses the `people` table where `Name` is stored because "people" wasn't in the question.
- **Over-selection:** If a token matches multiple tables, all are included. This rarely hurts accuracy compared to under-selection.

## Invalid SQL Failures (320 samples)

The model generated SQL that failed SQLite execution in 320 cases. These were caused by:

1. **Nonexistent Tables (Schema Retrieval)**
   - Consequence of linking failures. The model hallucinates generic tables like `users` or `countries`.
2. **Wrong Table/Column Association (Model Generation)**
   - The correct tables were in the prompt, but the model hallucinated relationships. 
   - *Example:* `no such column: T2.Content` when joining `TV_Channel` and `Cartoon`.
3. **Invalid SQL Constructs (Model Generation)**
   - *Example:* Using `YEAR(date)` which is invalid in SQLite (requires `strftime`).
   - *Example:* Using `pet_type` instead of the schema-provided `petType`.

*Conclusion:* Half of invalid SQL is due to empty schemas (linking failures). The other half is the model struggling with exact schema casing/aliases or hallucinating functions.

## Semantic SQL Failures (382 samples)

The model generated valid SQL, but it returned the wrong data in 382 cases. These are purely **reasoning/generation failures**. The schema was present, but the model failed to use it correctly.

1. **Missing JOINs:** Generating `SELECT People_ID FROM poker_player` instead of joining to `people` to get the actual names requested.
2. **Over-selecting Columns:** `SELECT * FROM Students` instead of selecting the specific detail requested.
3. **Incorrect Filters:** Missing implicit filters required by the domain.

## Aggregation / Reasoning Analysis

Questions requiring aggregations (SUM, AVG, MIN, MAX, GROUP BY, HAVING) form a massive failure mode.
- **Total Aggregation Questions:** 548 samples
- **Execution Accuracy on Aggregation:** 22.4% (vs 32.1% overall)

*Example Failure:*
- **Q:** "Which dogs have not cost their owner more than 1000..."
- **Gen SQL:** Uses a simple `JOIN` and `WHERE cost <= 1000`.
- **Gold SQL:** Uses `NOT IN (SELECT ... GROUP BY ... HAVING sum > 1000)`.
- *Conclusion:* The model fails to recognize when to use `HAVING SUM()` vs `WHERE`. This is a pure SQL reasoning failure.

## Conclusion

Based on this analysis, **schema retrieval is no longer the primary problem once NLP normalization is applied**. The remaining failures are reasoning errors: the model has the schema but doesn't know how to write the correct SQLite aggregations or JOINs. This motivates the need for retrieved few-shot SQL demonstrations to anchor the model's logic capabilities.
