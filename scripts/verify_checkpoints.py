"""Quick sanity check: confirm generated LoRA checkpoints are valid PEFT artifacts."""
from pathlib import Path

from peft import PeftConfig
from safetensors import safe_open


DIRS = [
    "checkpoints/phase2/platform_lora",
    "checkpoints/phase3/simulator_lora",
    "checkpoints/phase4/platform_lora",
    "checkpoints/last_stable/platform_lora",
    "checkpoints/last_stable/simulator_lora",
]


def main() -> None:
    for d in DIRS:
        cfg = PeftConfig.from_pretrained(d)
        with safe_open(str(Path(d) / "adapter_model.safetensors"), framework="pt") as f:
            keys = list(f.keys())
            first = keys[0]
            shape = f.get_tensor(first).shape
        print(
            f"[OK] {d}\n"
            f"       base      : {cfg.base_model_name_or_path}\n"
            f"       r         : {cfg.r}\n"
            f"       alpha     : {cfg.lora_alpha}\n"
            f"       targets   : {cfg.target_modules}\n"
            f"       tensors   : {len(keys)}\n"
            f"       first key : {first}\n"
            f"       first shp : {tuple(shape)}\n"
        )


if __name__ == "__main__":
    main()
