"""Write randomly-initialized adapter_model.safetensors into each checkpoint dir.

NOT a trained model — only valid LoRA weights of the right shape so PEFT/Unsloth
can load the directory without crashing. lora_A is Kaiming-uniform, lora_B is
zeros, mirroring PEFT's default fresh-LoRA init (effective delta = 0).

Use this when the trained .safetensors files are missing (e.g., after a clone
where they were gitignored) and you need the load path to work for tests/CI.
For real inference, retrain via run_training.sh.
"""

from __future__ import annotations

import json
import math
import struct
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent

# Qwen2.5 architecture sizes — pulled from the published HF configs.
# (hidden_size, num_layers, kv_out_dim) where kv_out_dim = num_kv_heads * head_dim.
QWEN_ARCH = {
    "Qwen/Qwen2.5-1.5B-Instruct": {
        "hidden_size": 1536,
        "num_layers":  28,
        "kv_out_dim":  256,   # 2 kv heads * 128 head_dim
    },
    "Qwen/Qwen2.5-0.5B-Instruct": {
        "hidden_size": 896,
        "num_layers":  24,
        "kv_out_dim":  128,   # 2 kv heads * 64 head_dim
    },
}

CHECKPOINT_DIRS = [
    REPO_ROOT / "checkpoints" / "phase2"       / "platform_lora",
    REPO_ROOT / "checkpoints" / "phase3"       / "simulator_lora",
    REPO_ROOT / "checkpoints" / "phase4"       / "platform_lora",
    REPO_ROOT / "checkpoints" / "last_stable"  / "platform_lora",
    REPO_ROOT / "checkpoints" / "last_stable"  / "simulator_lora",
]


def kaiming_uniform_(shape: tuple[int, int], rng: np.random.Generator) -> np.ndarray:
    """torch.nn.init.kaiming_uniform_(a=sqrt(5)) — PEFT's default for lora_A."""
    fan_in = shape[1]
    bound = math.sqrt(1.0 / fan_in)
    return rng.uniform(-bound, bound, size=shape).astype(np.float32)


def build_lora_tensors(arch: dict, rank: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Return {param_name: tensor} for a Qwen2.5 LoRA on q_proj + v_proj."""
    hidden = arch["hidden_size"]
    kv_out = arch["kv_out_dim"]
    tensors: dict[str, np.ndarray] = {}

    for layer in range(arch["num_layers"]):
        prefix = f"base_model.model.model.layers.{layer}.self_attn"

        # q_proj: out=hidden, in=hidden
        tensors[f"{prefix}.q_proj.lora_A.weight"] = kaiming_uniform_((rank, hidden), rng)
        tensors[f"{prefix}.q_proj.lora_B.weight"] = np.zeros((hidden, rank), dtype=np.float32)

        # v_proj: out=kv_out (GQA), in=hidden
        tensors[f"{prefix}.v_proj.lora_A.weight"] = kaiming_uniform_((rank, hidden), rng)
        tensors[f"{prefix}.v_proj.lora_B.weight"] = np.zeros((kv_out, rank), dtype=np.float32)

    return tensors


def write_safetensors(path: Path, tensors: dict[str, np.ndarray]) -> None:
    """Write a safetensors file by hand: 8-byte header length + JSON header + raw bytes."""
    header: dict[str, dict] = {}
    offset = 0
    for name, arr in tensors.items():
        nbytes = arr.nbytes
        header[name] = {
            "dtype": "F32",
            "shape": list(arr.shape),
            "data_offsets": [offset, offset + nbytes],
        }
        offset += nbytes

    header_json = json.dumps(header, separators=(",", ":")).encode("utf-8")
    # Pad header to 8-byte alignment (safetensors convention).
    pad = (8 - (len(header_json) % 8)) % 8
    header_json += b" " * pad

    with path.open("wb") as f:
        f.write(struct.pack("<Q", len(header_json)))
        f.write(header_json)
        for name in tensors:
            f.write(tensors[name].tobytes(order="C"))


def process_dir(ckpt_dir: Path) -> None:
    config_path = ckpt_dir / "adapter_config.json"
    if not config_path.exists():
        print(f"[SKIP] {ckpt_dir} — no adapter_config.json")
        return

    with config_path.open() as f:
        cfg = json.load(f)

    base_model = cfg["base_model_name_or_path"]
    rank = cfg["r"]
    arch = QWEN_ARCH.get(base_model)
    if arch is None:
        print(f"[SKIP] {ckpt_dir} — unknown base model {base_model}")
        return

    # Deterministic seed per-dir so reruns produce identical files.
    seed = abs(hash(str(ckpt_dir.relative_to(REPO_ROOT)))) % (2**32)
    rng = np.random.default_rng(seed)

    tensors = build_lora_tensors(arch, rank, rng)
    out_path = ckpt_dir / "adapter_model.safetensors"
    write_safetensors(out_path, tensors)

    total_bytes = sum(t.nbytes for t in tensors.values())
    print(
        f"[OK] {ckpt_dir.relative_to(REPO_ROOT)}  "
        f"base={base_model}  r={rank}  "
        f"params={sum(t.size for t in tensors.values()):,}  "
        f"size={total_bytes/1024:.1f} KiB"
    )


def main() -> None:
    print(f"Repo root: {REPO_ROOT}")
    print(f"Writing placeholder adapter_model.safetensors into {len(CHECKPOINT_DIRS)} dirs...\n")
    for d in CHECKPOINT_DIRS:
        process_dir(d)
    print("\nDone. These are placeholder weights (lora_B = zeros → effective delta = 0).")
    print("They make PEFT load paths work but produce base-model behavior at inference.")
    print("Re-run training (run_training.sh) to replace with real trained weights.")


if __name__ == "__main__":
    main()
