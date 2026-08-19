"""Experiment orchestration and result tracking."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from modelbench.config import Config
from modelbench.dataset import SpiderDataset
from modelbench.evaluation import evaluate_sample
from modelbench.extract import SQLExtractionError, extract_sql
from modelbench.model import create_model
from modelbench.prompt import build_text_to_sql_prompt
from modelbench.schema import extract_schema_from_db
from modelbench.types import ExperimentMetadata, ExperimentResult, SampleResult

logger = logging.getLogger(__name__)


class ExperimentRunner:
    """Orchestrates a Text-to-SQL experiment run."""

    def __init__(self, config: Config):
        """Initialize the experiment runner.

        Args:
            config: Full experiment configuration.
        """
        self.config = config

        # We only support Spider in M3
        if self.config.dataset.name != "spider":
            raise ValueError(f"Unsupported dataset: {self.config.dataset.name}")

        self.dataset = SpiderDataset(self.config.dataset)
        self._model = None

    @property
    def model(self):
        """Lazily load the model on first access."""
        if self._model is None:
            logger.info("Initializing model: %s", self.config.model.model_id)
            self._model = create_model(self.config.model, self.config.generation)
        return self._model

    def run(self) -> ExperimentResult:
        """Run the configured experiment.

        Returns:
            The aggregate experiment results.
        """
        logger.info("Starting experiment: %s", self.config.experiment.name)

        metadata = ExperimentMetadata(
            experiment_name=self.config.experiment.name,
            dataset=self.config.dataset.name,
            split=self.config.dataset.split,
            limit=self.config.dataset.limit,
            seed=self.config.dataset.seed or self.config.experiment.seed,
            model_id=self.config.model.model_id,
            model_revision=self.config.model.revision,
            schema_strategy=self.config.schema.strategy,
            prompting_strategy=self.config.strategy.name,
            generation_config={
                "max_new_tokens": self.config.generation.max_new_tokens,
                "temperature": self.config.generation.temperature,
                "do_sample": self.config.generation.do_sample,
            },
        )

        sample_results: list[SampleResult] = []

        # Keep track of schemas to avoid re-extracting for the same DB
        schema_cache: dict[Path, str] = {}

        for sample in self.dataset.load():
            db_id = sample.db_id
            db_path_obj = Path(sample.db_path)

            # 1. Schema Generation
            if self.config.schema.strategy != "full":
                raise ValueError(f"Unsupported schema strategy: {self.config.schema.strategy}")

            if db_path_obj not in schema_cache:
                try:
                    schema_cache[db_path_obj] = extract_schema_from_db(db_path_obj)
                except Exception as e:
                    logger.error("Failed to extract schema for %s: %s", db_path_obj, e)
                    # Proceed with empty schema if extraction fails to gracefully record failure
                    schema_cache[db_path_obj] = ""

            schema = schema_cache[db_path_obj]

            # 2. Prompt Building
            if self.config.strategy.name != "zero_shot":
                raise ValueError(f"Unsupported prompting strategy: {self.config.strategy.name}")

            prompt = build_text_to_sql_prompt(sample.question, schema)

            # 3. Generation
            gen_result = self.model.generate(prompt)

            # 4. SQL Extraction
            extracted_sql = None
            sql_valid = False
            exact_match = False
            exec_acc = False
            exec_err = None

            try:
                extracted_sql = extract_sql(gen_result.text)

                # 5. Evaluation
                eval_res = evaluate_sample(extracted_sql, sample.gold_sql, sample.db_path)
                sql_valid = eval_res.sql_valid
                exact_match = eval_res.exact_match
                exec_acc = eval_res.execution_accuracy
                exec_err = eval_res.execution_error
            except SQLExtractionError as e:
                exec_err = f"Extraction failed: {e}"
            except Exception as e:
                exec_err = f"Evaluation failed: {e}"

            sample_id = f"{db_id}_{len(sample_results)}"

            sample_results.append(
                SampleResult(
                    sample_id=sample_id,
                    db_id=db_id,
                    question=sample.question,
                    gold_sql=sample.gold_sql,
                    generated_text=gen_result.text,
                    extracted_sql=extracted_sql,
                    sql_valid=sql_valid,
                    exact_match=exact_match,
                    execution_accuracy=exec_acc,
                    execution_error=exec_err,
                    latency_seconds=gen_result.latency_seconds,
                    input_tokens=gen_result.input_tokens,
                    output_tokens=gen_result.output_tokens,
                )
            )

        total_samples = len(sample_results)
        valid_count = sum(1 for r in sample_results if r.sql_valid)
        exact_count = sum(1 for r in sample_results if r.exact_match)
        exec_count = sum(1 for r in sample_results if r.execution_accuracy)

        avg_latency = 0.0
        if total_samples > 0:
            avg_latency = sum(r.latency_seconds for r in sample_results) / total_samples

        return ExperimentResult(
            metadata=metadata,
            total_samples=total_samples,
            valid_sql_count=valid_count,
            exact_match_count=exact_count,
            execution_correct_count=exec_count,
            sql_validity_rate=(valid_count / total_samples) if total_samples else 0.0,
            exact_match_rate=(exact_count / total_samples) if total_samples else 0.0,
            execution_accuracy=(exec_count / total_samples) if total_samples else 0.0,
            avg_latency_seconds=avg_latency,
            samples=sample_results,
        )

    def save_result(self, result: ExperimentResult, output_dir: str | Path = "results") -> Path:
        """Save the experiment result to JSON."""
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        file_path = out_dir / f"{self.config.experiment.name}.json"

        # Convert frozen dataclass to dict via custom serialization if needed,
        # but since they only contain standard types, we can use built-in dataclass conversion.
        import dataclasses

        with file_path.open("w", encoding="utf-8") as f:
            json.dump(dataclasses.asdict(result), f, indent=2)

        logger.info("Saved experiment results to %s", file_path)
        return file_path
