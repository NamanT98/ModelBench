#!/usr/bin/env python
"""Real model smoke test for ModelBench M2.

This script downloads and runs Qwen/Qwen2.5-Coder-3B-Instruct against
the 5-question fixture benchmark.  It is NOT part of the automated test
suite and requires:

  - Internet access (to download the model on first run)
  - pip install 'modelbench[inference]'
  - Optional: a CUDA GPU with >= 6 GB VRAM

Usage:
    python scripts/smoke_test.py

If CUDA is unavailable the script falls back to CPU (slow but functional).
If dependencies are missing it prints a clear error and exits.
"""

from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path


def main() -> int:
    # ── Check dependencies ───────────────────────────────────────
    try:
        import torch
    except ImportError:
        print("ERROR: PyTorch is not installed.")
        print("Install with: pip install 'modelbench[inference]'")
        return 1

    try:
        import transformers  # noqa: F401
    except ImportError:
        print("ERROR: Hugging Face Transformers is not installed.")
        print("Install with: pip install 'modelbench[inference]'")
        return 1

    # ── Environment info ─────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"PyTorch version:   {torch.__version__}")
    print(f"Device:            {device}")
    if device == "cuda":
        print(f"GPU:               {torch.cuda.get_device_name(0)}")
        mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU memory:        {mem_gb:.1f} GB")
    else:
        print("WARNING: Running on CPU. Inference will be slow.")
    print()

    # ── Build model ──────────────────────────────────────────────
    from modelbench.config import GenerationConfig, ModelConfig
    from modelbench.evaluation import evaluate_sample
    from modelbench.extract import SQLExtractionError, extract_sql
    from modelbench.fixture import create_fixture_db, get_fixture_samples
    from modelbench.model import HuggingFaceCausalLM
    from modelbench.prompt import build_text_to_sql_prompt
    from modelbench.schema import extract_schema_from_db

    model_config = ModelConfig(
        model_id="Qwen/Qwen2.5-Coder-3B-Instruct",
        device="auto",
        dtype="auto",
    )
    gen_config = GenerationConfig(
        max_new_tokens=256,
        temperature=0.0,
        do_sample=False,
    )

    print("=" * 60)
    print("Model Configuration")
    print("=" * 60)
    print(f"  model_id:       {model_config.model_id}")
    print(f"  revision:       {model_config.revision}")
    print(f"  device:         {model_config.device}")
    print(f"  dtype:          {model_config.dtype}")
    print(f"  max_new_tokens: {gen_config.max_new_tokens}")
    print(f"  temperature:    {gen_config.temperature}")
    print(f"  do_sample:      {gen_config.do_sample}")
    print()

    model = HuggingFaceCausalLM(model_config, gen_config)

    # ── Prepare fixture ──────────────────────────────────────────
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = create_fixture_db(Path(tmpdir) / "fixture_ecommerce.db")
        samples = get_fixture_samples(db_path)
        schema = extract_schema_from_db(db_path)

        print("=" * 60)
        print("Fixture Evaluation (5 samples)")
        print("=" * 60)
        print()

        results = []
        latencies = []

        for i, sample in enumerate(samples, 1):
            prompt = build_text_to_sql_prompt(sample.question, schema)
            gen_result = model.generate(prompt)
            latencies.append(gen_result.latency_seconds)

            try:
                predicted_sql = extract_sql(gen_result.text)
            except SQLExtractionError as e:
                print(f"  [FAIL] {i}. {sample.question}")
                print(f"         Extraction failed: {e}")
                print(f"         Raw: {gen_result.text!r}")
                from modelbench.types import EvaluationResult

                results.append(
                    EvaluationResult(
                        sql_valid=False,
                        exact_match=False,
                        execution_accuracy=False,
                        execution_error=f"Extraction failed: {e}",
                    )
                )
                print()
                continue

            eval_result = evaluate_sample(predicted_sql, sample.gold_sql, sample.db_path)
            results.append(eval_result)

            status = "pass" if eval_result.execution_accuracy else "FAIL"
            print(f"  [{status}] {i}. {sample.question}")
            print(f"         Gold SQL:      {sample.gold_sql}")
            print(f"         Predicted SQL: {predicted_sql}")
            print(f"         Latency:       {gen_result.latency_seconds:.2f}s")
            print(f"         Tokens:        {gen_result.input_tokens} in, {gen_result.output_tokens} out")
            if eval_result.execution_error:
                print(f"         Error: {eval_result.execution_error}")
            print()

        # ── Summary ──────────────────────────────────────────────
        print("=" * 60)
        print("Summary")
        print("=" * 60)
        total = len(results)
        valid = sum(1 for r in results if r.sql_valid)
        exact = sum(1 for r in results if r.exact_match)
        exec_acc = sum(1 for r in results if r.execution_accuracy)
        avg_lat = sum(latencies) / len(latencies) if latencies else 0

        print(f"  Model:              {model_config.model_id}")
        print(f"  Device:             {device}")
        print(f"  Samples:            {total}")
        print(f"  SQL Validity:       {valid}/{total}")
        print(f"  Exact Match:        {exact}/{total}")
        print(f"  Execution Accuracy: {exec_acc}/{total}")
        print(f"  Avg Latency:        {avg_lat:.2f}s")
        print()
        print("NOTE: These are fixture smoke-test results from 5 questions,")
        print("      NOT benchmark performance numbers.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
