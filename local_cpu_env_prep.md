# Running this notebook locally on CPU

This repo's `Untitled.ipynb` (and the README's `requirements.txt` — note: not actually
present in the repo) assume a CUDA GPU (`device_map='cuda:0'`, `torch_dtype=torch.bfloat16`).
If your machine has no NVIDIA GPU, follow these steps instead.

## 1. Create a venv

```bash
cd /home/chucho/Projects/Antidote_experiments
python3 -m venv .venv
```

## 2. Install CPU-only PyTorch

Do **not** `pip install torch` plain — that can pull CUDA wheels (multi-GB, useless
without a GPU). Use PyTorch's CPU-only index instead:

```bash
.venv/bin/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## 3. Install the rest of the dependencies

```bash
.venv/bin/python -m pip install transformers peft datasets accelerate
```

## 4. Patch the notebook's model-loading cell for CPU

In the first cell of `Untitled.ipynb`, change:

```python
torch_dtype=torch.bfloat16,
device_map='cuda:0',
```

to:

```python
torch_dtype=torch.float32,
device_map='cpu',
```

(`bfloat16` on CPU is slow/unsupported on many machines; `float32` is the safe default.
`cuda:0` will raise immediately if there's no CUDA device.)

## 5. (Optional) Register the venv as a Jupyter kernel

Needed if you want to execute notebook cells in place (e.g. via VS Code, JupyterLab, or
`nbclient`) with this venv rather than your system/conda Python:

```bash
.venv/bin/python -m pip install ipykernel
.venv/bin/python -m ipykernel install --user --name antidote-venv --display-name "Antidote (.venv)"
```

Then select the **Antidote (.venv)** kernel in your notebook UI.

## 6. Run it

- First cell downloads `Qwen/Qwen2.5-0.5B` (~1GB) into `cache_dir/` on first run — expect
  a few minutes depending on your connection.
- Everything after runs on CPU, which is *much* slower than GPU (expect minutes instead of
  seconds for training/generation steps). Fine for smoke-testing the code path, not for
  real training runs.

## Notes

- No GPU is required for cells that only load/inspect the model or data (cells 1–4 in
  `Untitled.ipynb`). Actual DPO/tamper-resistance training (`training.py`) will be very
  slow on CPU and is not recommended for anything beyond a sanity check.
- If you hit HF rate limits, set `HF_TOKEN` in your environment (see README's
  "Required Environment Variables" section).
