"""Experiment orchestration and result tracking."""

from __future__ import annotations

import dataclasses
import json
import logging
from pathlib import Path
from tqdm import tqdm

from modelbench.config import Config, DatasetConfig
from modelbench.dataset import SpiderDataset
from modelbench.evaluation import evaluate_sample
from modelbench.extract import SQLExtractionError, extract_sql
from modelbench.model import create_model
from modelbench.prompt import build_text_to_sql_prompt
from modelbench.schema import create_schema_strategy, introspect_database
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

        # Instantiate the schema strategy from config
        self.schema_strategy = create_schema_strategy(
            self.config.schema.strategy,
            max_fk_depth=self.config.schema.max_fk_depth,
        )

        self.retriever = None
        if self.config.strategy.name == "few_shot":
            from modelbench.retrieval import create_retriever
            
            logger.info("Initializing few-shot retriever (strategy: %s)", self.config.strategy.retriever)
            train_config = DatasetConfig(
                name=self.config.dataset.name,
                path=self.config.dataset.path,
                split=self.config.strategy.train_split
            )
            train_dataset = SpiderDataset(train_config)
            train_samples = list(train_dataset.load())
            self.retriever = create_retriever(self.config.strategy.retriever, train_samples)

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

        # Cache DatabaseSchema domain objects per db_path
        schema_cache: dict[Path, object] = {}

        # Convert iterator to list for tqdm to have a total length
        samples_list = list(self.dataset.load())
        for sample in tqdm(samples_list, desc="Running Experiment", unit="sample"):
            db_id = sample.db_id
            db_path_obj = Path(sample.db_path)

            # 1. Schema Introspection (cached per database)
            if db_path_obj not in schema_cache:
                try:
                    schema_cache[db_path_obj] = introspect_database(db_path_obj)
                except Exception as e:
                    logger.error("Failed to introspect schema for %s: %s", db_path_obj, e)
                    schema_cache[db_path_obj] = None

            db_schema = schema_cache[db_path_obj]

            # 2. Schema Strategy → schema string
            schema_str = ""
            diag_dict = None
            if db_schema is not None:
                schema_str = self.schema_strategy.get_schema_string(db_schema, sample.question)
                diag = self.schema_strategy.get_diagnostics()
                diag_dict = dataclasses.asdict(diag)

            # 3. Prompt Building
            import time
            examples = None
            retrieval_diag = None
            if self.retriever is not None:
                start_retrieval = time.perf_counter()
                retrieved_samples = self.retriever.retrieve(sample.question, self.config.strategy.k)
                retrieval_latency = time.perf_counter() - start_retrieval
                
                examples = []
                for ex_sample in retrieved_samples:
                    ex_db_path_obj = Path(ex_sample.db_path)
                    if ex_db_path_obj not in schema_cache:
                        try:
                            schema_cache[ex_db_path_obj] = introspect_database(ex_db_path_obj)
                        except Exception as e:
                            logger.error("Failed to introspect schema for %s: %s", ex_db_path_obj, e)
                            schema_cache[ex_db_path_obj] = None
                            
                    ex_db_schema = schema_cache[ex_db_path_obj]
                    ex_schema_str = ""
                    if ex_db_schema is not None:
                        ex_schema_str = self.schema_strategy.get_schema_string(ex_db_schema, ex_sample.question)
                    examples.append((ex_sample, ex_schema_str))
                    
                retrieval_diag = {
                    "k": self.config.strategy.k,
                    "retrieved": len(retrieved_samples),
                    "latency_seconds": retrieval_latency,
                }

            prompt = build_text_to_sql_prompt(sample.question, schema_str, examples)

            # 4. Generation
            gen_result = self.model.generate(prompt)

            # 5. SQL Extraction & Evaluation
            extracted_sql = None
            sql_valid = False
            exact_match = False
            exec_acc = False
            exec_err = None

            try:
                extracted_sql = extract_sql(gen_result.text)

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
                    schema_diagnostics=diag_dict,
                    retrieval_diagnostics=retrieval_diag,
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

        with file_path.open("w", encoding="utf-8") as f:
            json.dump(dataclasses.asdict(result), f, indent=2)

        logger.info("Saved experiment results to %s", file_path)
        return file_path
