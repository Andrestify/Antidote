"""
Preflight per la Settimana 1 - riproduzione di AntiDote a piccola scala.

Da lanciare PRIMA di modificare training.py. Non addestra nulla: verifica
che le tre cose che possono far fallire silenziosamente il run siano a posto.

    python preflight.py --model_name Qwen/Qwen2.5-0.5B-Instruct \
                        --n_harmful 1000 --batch_size 2 --k_steps 2000
"""

import argparse
import torch
from transformers import AutoConfig


def derive_layer_configs(model_name: str) -> dict:
    """
    Restituisce il dizionario {nome_modulo: (in_features, out_features)} che
    Adversary(...) si aspetta, ricavandolo dalla config del modello invece di
    scriverlo a mano. Sostituisce layer_configs_3b in training.py.

    Attenzione a k_proj/v_proj: con Grouped-Query Attention l'output NON e'
    hidden_size ma num_key_value_heads * head_dim.
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
        print("  [!] Nessuna GPU visibile: il training non e' praticabile qui.")
        return
    name = torch.cuda.get_device_name(0)
    ok = torch.cuda.is_bf16_supported()
    print(f"  GPU: {name}")
    if ok:
        print("  bf16 supportato -> mixed_precision='bf16' va bene.")
    else:
        print("  [!] bf16 NON supportato (tipico di T4/P100 su Colab free).")
        print("      In training.py: Accelerator(mixed_precision='fp16').")
        print("      Occhio che con fp16 la loss DPO puo' andare in NaN:")
        print("      se succede, servono float32 sulle logprob o una GPU L4/A100.")


def check_blocks(n_harmful: int, batch_size: int, k_steps: int, epochs: int = 3) -> None:
    n_batches = n_harmful // batch_size
    num_blocks = n_batches // k_steps
    print(f"  esempi={n_harmful}  batch_size={batch_size}  -> {n_batches} batch")
    print(f"  k_steps={k_steps} -> num_blocks = {num_blocks}")
    if num_blocks == 0:
        suggested = max(1, n_batches // 4)
        print("  [!!] num_blocks = 0: il doppio ciclo NON parte e lo script")
        print("       arriva al merge senza errori, restituendo un modello")
        print(f"       non indurito. Abbassa k_steps a ~{suggested}.")
    else:
        total = epochs * num_blocks * 2 * k_steps
        print(f"  passi di ottimizzazione totali: {epochs} x {num_blocks} x 2 x {k_steps}"
              f" = {total:,}")
        if total > 5000:
            print("  [!] Troppi per Colab. Riduci k_steps, epoche o dataset.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--n_harmful", type=int, default=15580)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--k_steps", type=int, default=2000)
    p.add_argument("--epochs", type=int, default=3)
    args = p.parse_args()

    print("\n[1] layer_configs derivate dal modello")
    for k, v in derive_layer_configs(args.model_name).items():
        print(f"  '{k}': {v},")
    print("  -> incolla questo dict al posto di layer_configs_3b in training.py")

    print("\n[2] precisione numerica")
    check_bf16()

    print("\n[3] schedule k:k")
    check_blocks(args.n_harmful, args.batch_size, args.k_steps, args.epochs)
    print()
