"""Model abstraction for LLM inference.

Defines a lightweight :class:`Model` protocol and a concrete
:class:`HuggingFaceCausalLM` implementation.  All heavy dependencies
(``torch``, ``transformers``) are imported **lazily** so that CLI
commands like ``modelbench --help`` never trigger a multi-GB download.
"""

from __future__ import annotations

import logging
import os
import time

# Disable experimental vLLM V1 engine to prevent UVA errors on incompatible hardware
os.environ["VLLM_USE_V1"] = "0"
os.environ["VLLM_ENABLE_V1"] = "0"
os.environ["VLLM_WSL2_ENABLE_PIN_MEMORY"] = "1"
os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"

from typing import Protocol, runtime_checkable

from modelbench.config import GenerationConfig, ModelConfig
from modelbench.types import GenerationResult

logger = logging.getLogger(__name__)


# ── Model protocol ──────────────────────────────────────────────────


@runtime_checkable
class Model(Protocol):
    """Minimal interface for text-generation models.

    Any object that exposes :pyattr:`model_id` and :pymeth:`generate`
    with the correct signatures satisfies this protocol, including
    test fakes.
    """

    @property
    def model_id(self) -> str:
        """Return the model identifier (e.g. a Hugging Face repo ID)."""
        ...

    def generate(self, prompt: str) -> GenerationResult:
        """Generate text from a prompt.

        Args:
            prompt: The input text.

        Returns:
            A :class:`GenerationResult` with the generated text,
        """
        ...

    def generate_batch(self, prompts: list[str]) -> list[GenerationResult]:
        """Generate text from a batch of prompts.

        Args:
            prompts: The list of input texts.

        Returns:
            A list of GenerationResult objects corresponding exactly to the inputs.
        """
        ...


# ── Hugging Face implementation ─────────────────────────────────────


class HuggingFaceCausalLM:
    """Hugging Face ``AutoModelForCausalLM`` adapter.

    Model and tokenizer are loaded **lazily** on the first call to
    :meth:`generate`.  Construction is cheap and does not download
    anything.

    Args:
        model_config: Model identification and hardware settings.
        generation_config: Generation hyper-parameters.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        generation_config: GenerationConfig,
    ) -> None:
        self._model_config = model_config
        self._gen_config = generation_config
        self._tokenizer = None
        self._hf_model = None
        self._device: str | None = None

    # ── Protocol properties ──────────────────────────────────────

    @property
    def model_id(self) -> str:
        return self._model_config.model_id

    # ── Public API ───────────────────────────────────────────────

    def generate(self, prompt: str) -> GenerationResult:
        """Generate text from *prompt* using the configured model.

        On the first call the model and tokenizer are downloaded and
        loaded into memory (lazy initialisation).
        """
        self._ensure_loaded()
        assert self._tokenizer is not None
        assert self._hf_model is not None
        assert self._device is not None

        import torch  # already guaranteed importable after _ensure_loaded

        # Apply chat template if the tokenizer supports it
        if getattr(self._tokenizer, "chat_template", None):
            messages = [{"role": "user", "content": prompt}]
            input_text = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            input_text = prompt

        inputs = self._tokenizer(input_text, return_tensors="pt").to(self._device)
        input_token_count = inputs["input_ids"].shape[1]

        # Build generation kwargs
        gen_kwargs: dict = {
            "max_new_tokens": self._gen_config.max_new_tokens,
        }
        if self._gen_config.do_sample:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = self._gen_config.temperature
        else:
            gen_kwargs["do_sample"] = False

        start = time.perf_counter()
        with torch.no_grad():
            output_ids = self._hf_model.generate(**inputs, **gen_kwargs)
        elapsed = time.perf_counter() - start

        # Decode only the newly generated tokens
        new_ids = output_ids[0][input_token_count:]
        text = self._tokenizer.decode(new_ids, skip_special_tokens=True)

        return GenerationResult(
            text=text,
            latency_seconds=elapsed,
            output_tokens=len(new_ids),
        )

    def generate_batch(self, prompts: list[str]) -> list[GenerationResult]:
        """Generate text from a batch of prompts.
        
        The outputs are returned in exactly the same order as the inputs.
        Padding is applied to the left so that generation happens correctly.
        """
        if not prompts:
            return []

        self._ensure_loaded()
        assert self._tokenizer is not None
        assert self._hf_model is not None
        assert self._device is not None

        import torch

        # Ensure padding tokens are set appropriately for batching
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._tokenizer.padding_side = "left"

        input_texts = []
        if getattr(self._tokenizer, "chat_template", None):
            for p in prompts:
                messages = [{"role": "user", "content": p}]
                input_texts.append(
                    self._tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                )
        else:
            input_texts = prompts

        inputs = self._tokenizer(input_texts, return_tensors="pt", padding=True).to(self._device)
        input_token_count = inputs["input_ids"].shape[1]

        gen_kwargs: dict = {
            "max_new_tokens": self._gen_config.max_new_tokens,
        }
        if self._gen_config.do_sample:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = self._gen_config.temperature
        else:
            gen_kwargs["do_sample"] = False

        start = time.perf_counter()
        with torch.no_grad():
            output_ids = self._hf_model.generate(**inputs, **gen_kwargs)
        elapsed = time.perf_counter() - start

        results = []
        for i in range(len(prompts)):
            # Decode only the newly generated tokens
            new_ids = output_ids[i][input_token_count:]
            text = self._tokenizer.decode(new_ids, skip_special_tokens=True)
            
            # The true input token count for this specific sample before padding
            actual_input_tokens = int(inputs["attention_mask"][i].sum().item())
            
            results.append(
                GenerationResult(
                    text=text,
                    latency_seconds=elapsed,
                    input_tokens=actual_input_tokens,
                    output_tokens=len(new_ids),
                )
            )

        del inputs
        del output_ids
        torch.cuda.empty_cache()

        return results

    # ── Lazy loading internals ───────────────────────────────────

    def _ensure_loaded(self) -> None:
        """Download and load the model if not already done."""
        if self._hf_model is not None:
            return

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:
            raise ImportError(
                "Hugging Face Transformers is required for model inference. "
                "Install with: pip install 'modelbench[inference]'"
            ) from e

        try:
            import torch
        except ImportError as e:
            raise ImportError(
                "PyTorch is required for model inference. "
                "Install with: pip install 'modelbench[inference]'"
            ) from e

        device = self._resolve_device(torch)
        dtype = self._resolve_dtype(torch, device)

        logger.info(
            "Loading tokenizer: %s (revision: %s)",
            self._model_config.model_id,
            self._model_config.revision,
        )
        self._tokenizer = AutoTokenizer.from_pretrained(
            self._model_config.model_id,
            revision=self._model_config.revision,
        )

        logger.info(
            "Loading model: %s (dtype: %s, device: %s)",
            self._model_config.model_id,
            dtype,
            device,
        )
        self._hf_model = AutoModelForCausalLM.from_pretrained(
            self._model_config.model_id,
            revision=self._model_config.revision,
            torch_dtype=dtype,
        ).to(device)
        self._hf_model.eval()

        self._device = device
        logger.info("Model loaded successfully")

    def _resolve_device(self, torch_module: object) -> str:
        """Map the configured device string to a concrete device."""
        import torch

        if self._model_config.device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info("Auto-detected device: %s", device)
            return device
        return self._model_config.device

    def _resolve_dtype(self, torch_module: object, device: str) -> object:
        """Map the configured dtype string to a torch dtype."""
        import torch

        dtype_map = {
            "float16": torch.float16,
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
        }
        if self._model_config.dtype == "auto":
            resolved = torch.float16 if device == "cuda" else torch.float32
            logger.info("Auto-selected dtype: %s", resolved)
            return resolved
        return dtype_map[self._model_config.dtype]


# ── vLLM implementation ──────────────────────────────────────────────


class VLLMModel:
    """vLLM adapter for high-throughput inference.

    Model is loaded **lazily** on the first call to :meth:`generate` or :meth:`generate_batch`.

    Args:
        model_config: Model identification and hardware settings.
        generation_config: Generation hyper-parameters.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        generation_config: GenerationConfig,
    ) -> None:
        self._model_config = model_config
        self._gen_config = generation_config
        self._llm = None
        self._tokenizer = None

    @property
    def model_id(self) -> str:
        return self._model_config.model_id

    def generate(self, prompt: str) -> GenerationResult:
        """Generate text from a single prompt."""
        results = self.generate_batch([prompt])
        return results[0]

    def generate_batch(self, prompts: list[str]) -> list[GenerationResult]:
        """Generate text from a batch of prompts using vLLM."""
        if not prompts:
            return []

        self._ensure_loaded()
        assert self._llm is not None
        assert self._tokenizer is not None

        from vllm import SamplingParams

        # Apply chat template if the tokenizer supports it
        input_texts = []
        if getattr(self._tokenizer, "chat_template", None):
            for p in prompts:
                messages = [{"role": "user", "content": p}]
                input_texts.append(
                    self._tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                )
        else:
            input_texts = prompts

        sampling_params = SamplingParams(
            max_tokens=self._gen_config.max_new_tokens,
            temperature=self._gen_config.temperature if self._gen_config.do_sample else 0.0,
        )

        start = time.perf_counter()
        
        # vLLM automatically handles batching and returns outputs in the same order
        outputs = self._llm.generate(input_texts, sampling_params, use_tqdm=False)
        
        elapsed = time.perf_counter() - start

        results = []
        for output in outputs:
            text = output.outputs[0].text
            actual_input_tokens = len(output.prompt_token_ids)
            output_tokens = len(output.outputs[0].token_ids)
            
            results.append(
                GenerationResult(
                    text=text,
                    latency_seconds=elapsed,
                    input_tokens=actual_input_tokens,
                    output_tokens=output_tokens,
                )
            )

        return results

    def _ensure_loaded(self) -> None:
        """Load the vLLM engine if not already done."""
        if self._llm is not None:
            return

        try:
            from vllm import LLM
            from transformers import AutoTokenizer
        except ImportError as e:
            raise ImportError(
                "vLLM is required for the vllm provider. "
                "Install with: pip install vllm"
            ) from e

        logger.info(
            "Loading tokenizer: %s (revision: %s)",
            self._model_config.model_id,
            self._model_config.revision,
        )
        self._tokenizer = AutoTokenizer.from_pretrained(
            self._model_config.model_id,
            revision=self._model_config.revision,
        )

        logger.info(
            "Loading vLLM model: %s (gpu_memory_utilization: %s)",
            self._model_config.model_id,
            self._model_config.gpu_memory_utilization,
        )
        
        self._llm = LLM(
            model=self._model_config.model_id,
            revision=self._model_config.revision,
            gpu_memory_utilization=self._model_config.gpu_memory_utilization,
            tensor_parallel_size=1,
            trust_remote_code=True,
            enforce_eager=True,
            max_model_len=2048,
        )

        logger.info("vLLM Model loaded successfully")


# ── Factory ─────────────────────────────────────────────────────────


def create_model(
    model_config: ModelConfig,
    generation_config: GenerationConfig,
) -> Model:
    """Create a model instance from configuration.

    Currently only the ``"huggingface"`` provider is supported.

    Args:
        model_config: Model settings.
        generation_config: Generation settings.

    Returns:
        An object satisfying the :class:`Model` protocol.

    Raises:
        ValueError: If the provider is unsupported.
    """
    if model_config.provider == "huggingface":
        return HuggingFaceCausalLM(model_config, generation_config)
    elif model_config.provider == "vllm":
        return VLLMModel(model_config, generation_config)
    raise ValueError(f"Unsupported model provider: {model_config.provider!r}")
