# ModelBench V1 Validation Report

This document records the final validation checks performed before releasing ModelBench V1. It verifies reproducibility, testing integrity, and the equivalence of the M7 candidate pool optimization.

## 1. Reproducibility & Environment Validation

### Config Loading Checks
A Python script (`verify_configs.py`) was run against all non-experimental YAML configurations in the `configs/` directory to ensure they can be loaded, parsed, and instantiated without runtime errors.
- `configs/m4_b1_linking.yaml`: **PASS**
- `configs/m5_few_shot_k3.yaml`: **PASS**
- `configs/m6_embedding_k3.yaml`: **PASS**
- `configs/spider_qwen_hybrid_rrf_v1.yaml` (Final V1 Config): **PASS**

### Dependency Checks
- Initial experimentation with `vLLM` infrastructure was abandoned due to hardware/storage constraints. All vLLM dependencies, tests, and configuration references have been successfully excised from the V1 release.
- The default `batch_size` in the final `spider_qwen_hybrid_rrf_v1.yaml` configuration is correctly set to `8`.

## 2. Test Suite Validation

The full test suite (`pytest tests/ -v`) executes successfully locally on the `modelbench` conda environment.
- **Total Tests**: 140
- **Status**: **PASS** (100%)

Key test groupings that passed:
- `test_hybrid_retrieval.py`: Verified equivalence of bounded vs exhaustive RRF ranking.
- `test_retrieval.py`: Verified zero test set leakage into the training retrieval corpus.
- `test_schema.py`: Verified that M4-B.1 NLP-normalized schema linking accurately connects tables via Foreign Keys.
- `test_model.py`: Verified Hugging Face generation loop stability and deterministic text extraction.

## 3. M7 Candidate Pool Equivalence

The most significant CPU retrieval optimization in V1 was bounding the hybrid retrieval candidate pool (`candidate_n`). Our final validation tested `candidate_n=25` across all 1,034 samples of the Spider dev set.

**Equivalence Proof:**
- **Execution Accuracy (Full RRF)**: 58.03%
- **Execution Accuracy (Bounded N=25)**: 58.03%
- **Top-3 Retrieved Identity Rate**: 100%

Bounding the pool to $N=25$ produces *exactly* the same Top-3 retrieved few-shot examples as exhaustive full-corpus searching on this dataset, guaranteeing zero regression in model performance while significantly reducing CPU retrieval time.

## 4. Repository Cleanliness

A final audit was performed to ensure the repository is clean for GitHub tracking:
- Temporary benchmark scripts have been deleted.
- Cache files (`.jaccard_cache.pkl`, `.embedding_cache_v1.npy`, etc.) are explicitly ignored by git.
- The `results/` folder is ignored by git to prevent massive JSON blobs from bloating the repository history.
- Obsolete hyperparameter tuning configurations were successfully relocated to `configs/experimental/`.
