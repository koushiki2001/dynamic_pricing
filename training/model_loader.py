"""Unsloth-backed loaders for the platform and simulator LLMs.

Two roles, two model sizes:

  Platform LLM  -> Qwen2.5-1.5B-Instruct  (richer reasoning, 23 obs fields)
  Simulator LLM -> Qwen2.5-0.5B-Instruct  (binary bluff/honest decision)

Both load in 4-bit via Unsloth's FastLanguageModel and are wrapped with LoRA
adapters when freshly loaded. Pass an explicit ``lora_path`` to resume from a
saved checkpoint (Phase 3+).

Importing this module does NOT require Unsloth — the import is deferred into the
loader functions so the rest of the project (env server, tests, prompt builders)
keeps working on machines without a CUDA build.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

PLATFORM_BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
SIMULATOR_BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

PLATFORM_MAX_SEQ_LEN = 1024
SIMULATOR_MAX_SEQ_LEN = 512

PLATFORM_LORA_R = 16
SIMULATOR_LORA_R = 8


def _load_with_unsloth(
    base_model: str,
    lora_path: Optional[str],
    max_seq_length: int,
    lora_r: int,
) -> Tuple[object, object]:
    """Shared 4-bit loader. Defers the unsloth import so this file is safe
    to import on a machine without a CUDA toolchain."""
    from unsloth import FastLanguageModel  # noqa: WPS433  -- intentional lazy import

    if lora_path is not None and Path(lora_path).exists():
        # Resume from a saved LoRA. Unsloth stores the base model name + adapter
        # weights together, so passing the LoRA path directly is enough.
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=lora_path,
            max_seq_length=max_seq_length,
            load_in_4bit=True,
        )
        return model, tokenizer

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_r,
        target_modules=["q_proj", "v_proj"],
        lora_alpha=lora_r,
        lora_dropout=0.0,
        bias="none",
    )
    return model, tokenizer


def enable_inference_mode(model) -> object:
    """Call FastLanguageModel.for_inference() for ~2x faster generation speed.

    Use this on any model that will only be doing inference (frozen opponent,
    eval runs, demo). Do NOT call on a model you still intend to train —
    it disables gradient computation.
    """
    try:
        from unsloth import FastLanguageModel
        return FastLanguageModel.for_inference(model)
    except Exception:
        model.eval()
        return model


def load_platform_model(lora_path: Optional[str] = None) -> Tuple[object, object]:
    """Load the platform LLM. Pass ``lora_path`` to resume from a checkpoint."""
    return _load_with_unsloth(
        base_model=PLATFORM_BASE_MODEL,
        lora_path=lora_path,
        max_seq_length=PLATFORM_MAX_SEQ_LEN,
        lora_r=PLATFORM_LORA_R,
    )


def load_simulator_model(lora_path: Optional[str] = None) -> Tuple[object, object]:
    """Load the simulator LLM. Pass ``lora_path`` to resume from a checkpoint."""
    return _load_with_unsloth(
        base_model=SIMULATOR_BASE_MODEL,
        lora_path=lora_path,
        max_seq_length=SIMULATOR_MAX_SEQ_LEN,
        lora_r=SIMULATOR_LORA_R,
    )
