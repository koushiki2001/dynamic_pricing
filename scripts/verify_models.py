"""Phase 1 sanity check: load both models, run a short generation each.

Run this on the GPU training machine after `pip install trl unsloth ...`.

  python scripts/verify_models.py
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from training.model_loader import load_platform_model, load_simulator_model


def _gen(model, tokenizer, prompt: str, max_new_tokens: int = 32) -> str:
    import torch  # local import — only needed on the training box

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def main() -> None:
    print("[VERIFY] Loading platform model (Qwen2.5-1.5B-Instruct)...")
    t0 = time.time()
    pm, pt = load_platform_model()
    print(f"[VERIFY] Platform loaded in {time.time()-t0:.1f}s")

    print("[VERIFY] Loading simulator model (Qwen2.5-0.5B-Instruct)...")
    t0 = time.time()
    sm, st = load_simulator_model()
    print(f"[VERIFY] Simulator loaded in {time.time()-t0:.1f}s")

    print("\n[VERIFY] Platform generation test:")
    out = _gen(pm, pt, 'Respond with JSON only: {"price": <number>}\nWhat price?')
    print(f"  -> {out!r}")

    print("\n[VERIFY] Simulator generation test:")
    out = _gen(sm, st, 'Respond with JSON only: {"rider": "accept" or "reject"}\nDecision?')
    print(f"  -> {out!r}")

    print("\n[VERIFY] Both models loaded and generating. Phase 1 model setup OK.")


if __name__ == "__main__":
    main()
