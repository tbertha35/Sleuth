# Sleuth

A fully local lab for building and understanding LLMs:

- **FORGE**: our own tokenizer, model, training loop and inference engine, in plain PyTorch.
- **ANALYZER**: a "brain viewer" for activations, attention, MoE routing, ablations,
  logit lens, and how they all change across training checkpoints.
- **LAB**: equal-compute experiments for making small models faster and act bigger.
- **ASSISTANT** (optional, off by default): explains results via OpenAI, Anthropic,
  llama.cpp or vLLM.

Nothing touches the network unless you turn it on. Design: [PLAN.md](PLAN.md) ·
decisions and results: [LOG.md](LOG.md) · training data and licences: [docs/DATA.md](docs/DATA.md)

![Dashboard](docs/img/dashboard.png)

## Status

**Milestone 1 (tokenizer → 10M model → checkpoints → generation → first analyzer view)
is built and tested on CPU.** The first GPU run happens on your machines (steps below).

| Piece | Where |
|---|---|
| Synthetic corpus generator | `sleuth/forge/data/synthetic.py` |
| Byte-level BPE (train / encode / decode / save) | `sleuth/forge/tokenizer/` |
| Ingest + memory-mapped shards + loader | `sleuth/forge/data/` |
| Transformer (RoPE, RMSNorm, SwiGLU, GQA, QK-norm, tying, taps) | `sleuth/forge/model/` |
| Trainer (bf16, grad accumulation, cosine/WSD, snapshots, exact resume) | `sleuth/forge/train/` |
| Generation (KV cache, temperature/top-k/top-p/min-p, tok/s) | `sleuth/forge/infer/` |
| Our own `.safetensors` reader/writer | `sleuth/formats/safetensors.py` |
| Analyzer: neuron stats across checkpoints | `sleuth/analyzer/` |
| Local dashboard | `sleuth/ui/` |
| Hardware probe | `scripts/probe_hardware.py` |

## Setup (each machine)

```bash
git clone https://github.com/tbertha35/Sleuth.git && cd Sleuth
git checkout claude/quirky-hamilton-fst7bf       # until it's merged
python3 -m venv .venv && source .venv/bin/activate   # Python 3.11+; 3.12 is the safest bet
```

Install PyTorch for the machine, **then** Sleuth:

| Machine | PyTorch install |
|---|---|
| RTX 5060 Ti / RTX 3060 Ti (x86_64) | `pip install torch` (needs an NVIDIA driver new enough for its CUDA version; if `torch.cuda.is_available()` is False, try `pip install torch --index-url https://download.pytorch.org/whl/cu128`) |
| GX10 / GB10 (aarch64, sm_121) | PyPI has no CUDA build for this. Use `pip install torch --index-url https://download.pytorch.org/whl/cu130`, or NVIDIA's PyTorch container (NGC). Then run the probe: it checks that the build has Blackwell kernels and that each attention kernel actually runs (some have failed on sm_121). |

**Windows (the RTX boxes):** create the venv with `py -3.12 -m venv .venv` and activate it
with `.venv\Scripts\Activate.ps1` (PowerShell). Everything else is the same. Differences
you'll see:
- PyTorch's Windows builds have no FlashAttention kernel, so attention falls back to
  the memory-efficient, cuDNN or plain kernels. That's fine for small models, and the
  probe records which ones work.
- `torch.compile` needs Triton, which isn't bundled on Windows, so keep `compile = false`.
  (A community `triton-windows` package exists; it would be a new dependency, so ask first.)
- No NCCL, so no multi-GPU training on Windows. That only matters on the GX10s (Linux).
- Environment variables in PowerShell: `$env:SLEUTH_DATA = "D:\sleuth\data"`.
- WSL2 (Linux on Windows) would get FlashAttention and Triton. It's optional and can come later.

```bash
pip install -e ".[dev]"        # numpy + pytest, and the `sleuth` command
pytest -q                      # ~80 CPU tests, ~10 s
python scripts/probe_hardware.py --disk-path /path/to/your/nvme   # saves hardware/<host>.json (stays local)
```

The `hardware/*.json` files are git-ignored, so they never leave your machine.
Paste the printed summary to Claude instead; anonymized numbers go into `LOG.md`.

## First run (Milestone 1)

```bash
# The whole pipeline on CPU in ~1 minute: synthetic data -> tokenizer -> shards -> train -> analyze
python -m sleuth demo

# The real M1 model on a GPU (~10M params; minutes on a 5060 Ti or GB10).
# On synthetic data it overfits after ~2000 steps (too little data); see the TinyStories run below.
python -m sleuth info configs/tiny-10m.toml             # params, FLOPs, token budget
python -m sleuth train configs/tiny-10m.toml --set data.shards=data/shards/synthetic
python -m sleuth generate runs/<run-id> --prompt "Once upon a time"
python -m sleuth analyze runs/<run-id>
python -m sleuth ui                                      # http://127.0.0.1:8765
```

Then the TinyStories model (download and prepare the data first: see `docs/DATA.md`):

```bash
python -m sleuth train configs/tinystories-10m.toml
python -m sleuth generate runs/<run-id> --prompt "Once upon a time" --samples 3
```

Training writes `runs/<run-id>/` with the config, the environment, metrics, weight
snapshots (log-spaced: steps 0, 1, 2, 4, … plus every 500) and resumable checkpoints.
Ctrl-C saves a checkpoint. Continue with `python -m sleuth train --resume runs/<run-id>`.

Things to try:
- `python -m sleuth tokenize --tokenizer data/tokenizers/synthetic-1k.json "any text"`
  shows how text is split into tokens.
- In the dashboard, set the metric to "Category-selective neurons" and watch neurons
  specialize in stories vs. math vs. code as training goes on.
- Generate from early snapshots (`--step 16`, `--step 128`) to see grammar, then
  names, then arithmetic emerge.

Where data and runs live: `data/` and `runs/` in the repo by default. To put them on
another drive, set `SLEUTH_DATA=/mnt/fast/sleuth-data` and `SLEUTH_RUNS=/mnt/fast/sleuth-runs`.

## Tests

```bash
pytest -q          # CPU tests
pytest -m gpu -q   # GPU-only tests (none yet; they arrive with M2)
```
