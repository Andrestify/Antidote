"""
Preflight for Week 1 - small-scale reproduction of AntiDote.

Run this BEFORE modifying training.py. It trains nothing: it checks that the
three things that can make a run fail silently are in order.

    python preflight.py --model_name Qwen/Qwen2.5-0.5B-Instruct \
                        --n_harmful 1000 --batch_size 2 --k_steps 2000
"""

import argparse
import torch
from transformers import AutoConfig


def derive_layer_configs(model_name: str) -> dict:
    """
    Returns the {module_name: (in_features, out_features)} dict that
    Adversary(...) expects, deriving it from the model config instead of
    hardcoding it. Replaces layer_configs_3b in training.py.

    Watch out for k_proj/v_proj: with Grouped-Query Attention the output is
    NOT hidden_size but num_key_value_heads * head_dim.
    """
    cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    hidden = cfg.hidden_size
    n_heads = cfg.num_attention_heads
    n_kv = getattr(cfg, "num_key_value_heads", n_heads)
    head_dim = getattr(cfg, "head_dim", hidden // n_heads)
    kv_out = n_kv * head_dim

    return {
        "q_proj": (hidden, n_heads * head_dim),
        "o_proj": (n_heads * head_dim, hidden),
        "k_proj": (hidden, kv_out),
        "v_proj": (hidden, kv_out),
    }


def check_bf16() -> None:
    if not torch.cuda.is_available():
        print("  [!] No GPU visible: training is not feasible here.")
        return
    name = torch.cuda.get_device_name(0)
    ok = torch.cuda.is_bf16_supported()
    print(f"  GPU: {name}")
    if ok:
        print("  bf16 supported -> mixed_precision='bf16' is fine.")
    else:
        print("  [!] bf16 NOT supported (typical of T4/P100 on free Colab).")
        print("      In training.py: Accelerator(mixed_precision='fp16').")
        print("      Beware that with fp16 the DPO loss can go NaN:")
        print("      if that happens, you need float32 logprobs or an L4/A100 GPU.")


def check_blocks(n_harmful: int, batch_size: int, k_steps: int, epochs: int = 3) -> None:
    n_batches = n_harmful // batch_size
    num_blocks = n_batches // k_steps
    print(f"  examples={n_harmful}  batch_size={batch_size}  -> {n_batches} batches")
    print(f"  k_steps={k_steps} -> num_blocks = {num_blocks}")
    if num_blocks == 0:
        suggested = max(1, n_batches // 4)
        print("  [!!] num_blocks = 0: the bi-level loop does NOT run and the")
        print("       script reaches the merge without errors, returning an")
        print(f"       un-hardened model. Lower k_steps to ~{suggested}.")
    else:
        total = epochs * num_blocks * 2 * k_steps
        print(f"  total optimization steps: {epochs} x {num_blocks} x 2 x {k_steps}"
              f" = {total:,}")
        if total > 5000:
            print("  [!] Too many for Colab. Reduce k_steps, epochs or dataset size.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--n_harmful", type=int, default=15580)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--k_steps", type=int, default=2000)
    p.add_argument("--epochs", type=int, default=3)
    args = p.parse_args()

    print("\n[1] layer_configs derived from the model")
    for k, v in derive_layer_configs(args.model_name).items():
        print(f"  '{k}': {v},")
    print("  -> paste this dict in place of layer_configs_3b in training.py")

    print("\n[2] numerical precision")
    check_bf16()

    print("\n[3] k:k schedule")
    check_blocks(args.n_harmful, args.batch_size, args.k_steps, args.epochs)
    print()
