# Sleuth: Plan

A fully local lab for building LLMs from scratch (**FORGE**), looking inside them
while they run (**ANALYZER**), and running controlled experiments to make small
models faster and better (**LAB**). An optional **ASSISTANT** explains things
along the way.

> Status: **M0 and M1 done** (M1 trained on an RTX 5060 Ti); **M3 (analyzer core)
> built** and run on a 10M TinyStories model and Qwen3-0.6B; **M5 (external models)
> in progress** (Qwen done; Gemma 4 runs and matches llama.cpp, but slowly). M2 (multi-GPU,
> 125M) waits on the GX10s being set up. Sections marked ❓ still need a decision from you
> (collected in [Open questions](#10-open-questions-for-you)).

---

## 0. Read this first: advice and corrections

You said you're learning and might have missed things. These are the points
that would change the plan, roughly in order of importance.

### 0.1 The GB10's memory is unified, so "GPU tier vs host-RAM tier" collapses there
On the GX10 (GB10), CPU and GPU share one 128 GB LPDDR5x pool. Published
bandwidth is about 273 GB/s; the probe will measure the real number. That means:

- On the **GX10s**, the GPU and host-RAM tiers of the Engram table are the
  *same physical memory*. They still differ in *how* you access them (a GPU
  gather kernel, or a CPU gather plus a copy), but you won't see the classic
  "VRAM fast, RAM slow" cliff. The tier that matters there is **RAM → NVMe**.
- On the **RTX 5060 Ti box** (16 GB GDDR7 over PCIe), all three tiers are real
  and the gaps between them are large. **That box is the better testbed for
  the tier experiment**, and the GB10 is the better testbed for *huge* tables.
- **Bandwidth limits decode speed, not compute.** Generating one token at
  batch size 1 reads every weight once. A 1B model in bf16 is 2 GB, so the GB10
  tops out around 273/2 ≈ 135 tok/s. The 5060 Ti (~448 GB/s) is *faster* for
  small-model inference, even though the GB10 has far more compute. Expect this
  surprise, and it's why quantization matters for inference (§6, item 5).

### 0.2 Compute budget: 1B is a multi-week run, 350M is a few days
Rule of thumb: training FLOPs ≈ 6 × params × tokens, and a "compute-optimal"
(Chinchilla) run uses ~20 tokens per parameter. The figures below assume
~40 TFLOP/s *achieved* bf16 per GB10, a placeholder until the probe gives us
the real peak and we measure real MFU (model FLOPs utilization).

| Model | Tokens (20×) | FLOPs | 1 node | 2 nodes (~1.8×) | Text needed (~4 bytes/token) |
|---|---|---|---|---|---|
| 10M | 0.2B (we'll do ~1B) | 6e16 | < 1 h | n/a | ~1–4 GB |
| 125M | 2.5B | 1.9e18 | ~13 h | ~7 h | ~10 GB |
| 350M | 7B | 1.5e19 | ~4 days | ~2.3 days | ~28 GB |
| 1B | 20B | 1.2e20 | ~35 days | ~19 days | ~80 GB |

Takeaways:
- **Do you have ~30–80 GB of good text?** ❓ That, more than hardware, may cap
  the largest useful model. Training past 20 tokens/param is fine (modern
  small models use 100×+), but *repeating* data more than ~4 epochs hurts.
- **The LAB wants many cheap runs, not one big one.** Most experiments should
  happen at 10M–125M, where a run takes minutes to hours and you can afford
  baselines and seeds. Then confirm the winners at 350M.
- **Two nodes are often better used as two independent experiments** than as
  one 2-node job. We'll support both; multi-node is for the big runs.

### 0.3 Vocabulary size dominates a 10M model
With `d_model=256` and a 32k vocab, the embedding table alone is 8.2M
parameters, nearly the whole "10M" model. Two fixes:
- Count **non-embedding parameters** when we say "10M", as most papers do, and
  report both counts.
- Use a **smaller vocab (~8k) for the smoke-test model**, and 32k for 125M and up.

Comparing models with different tokenizers needs a tokenizer-independent
metric. We'll report **bits per byte** (validation loss normalized by raw text
bytes) everywhere, never just "loss".

### 0.4 "Equal compute" needs a precise definition, and Engram makes it tricky
Engram tables add *lots of parameters but almost no FLOPs*. So "Engram model
beats baseline at equal FLOPs" is true but partly unfair. Every LAB result will
report all three axes (**training FLOPs, total params, active params**) and
compare against two baselines:
1. **iso-FLOP**: same compute, smaller param count. This is the headline number.
2. **iso-param**: a dense model with the same *total* parameters. This is the
   stronger, more honest baseline.

Looped layers have the opposite problem: they add FLOPs without params. Those
get compared against an unshared model of equal depth too.

### 0.5 Noise is the enemy at small scale
At 10M–125M, seed-to-seed variance can be as large as the effect you're trying
to measure. The LAB harness will first measure the **noise floor** (the same
baseline run with 3–5 seeds) and flag any result smaller than ~2σ as "not
distinguishable". This is the main way to stay "rigorous and honest".

### 0.6 Small models need targeted evals, not just perplexity
Without downloading benchmarks, "does it act bigger?" needs local tests that
small models can actually show differences on. Recommend we write **synthetic
probe tasks** ourselves (no downloads needed):
- induction / copying, associative recall (Engram *should* help here)
- multi-digit arithmetic and multi-step tasks (looped depth *should* help here)
- bracket matching, simple code completion

Plus your own category test suites, and validation bits-per-byte. If you
already have local copies of standard benchmarks (HellaSwag, ARC, GSM8K…) as
jsonl, we can read them too. ❓

### 0.7 Pure-Python BPE: training is fine, encoding GBs is the slow part
- Training works on a *word-frequency table* (unique pre-tokenized chunks with
  counts), which is far smaller than the corpus, plus incremental pair-count
  updates. That's fast enough in Python.
- Counting the words and *encoding* the corpus touches every byte. We'll use
  `multiprocessing` plus a per-word encoding cache. Profile first; a C
  extension is the planned fallback (it needs only a C compiler, no new
  Python packages).
- Gotcha: GPT-style pre-tokenizer regexes use `\p{L}` (Unicode letter classes),
  which Python's built-in `re` **does not support** (it needs the third-party
  `regex` package). We'll write an equivalent pattern using `[^\W\d_]` for
  "letter", so we stay within stdlib.

### 0.8 Loading other models needs more than weight readers
Reading `.safetensors`/GGUF gives you tensors. To *run* those models in the
analyzer you also need:
- **an architecture adapter**: mapping their tensor names and quirks onto our
  model. The Llama family (Llama 3, Mistral, Qwen2/3) maps almost 1:1 onto
  RoPE + RMSNorm + SwiGLU + GQA, so we design our model to cover them.
  Qwen3-MoE and Mixtral give you real MoE routing to analyze.
- **their tokenizer**: GGUF embeds the vocab and merges; safetensors models
  ship a `tokenizer.json`. We write our own reader for the BPE variants.
  SentencePiece-style tokenizers (Llama 2, Gemma) are harder ❓.
- **GGUF dequantization**: quantized blocks (Q4_K, Q6_K, Q8_0, …) must be
  unpacked. We'll implement the common types in NumPy/PyTorch first. The
  "IQ" importance-quantized types are complex, so they come later.

### 0.9 Hugging Face downloads and Parquet
Many HF datasets are published only as **Parquet**, and reading Parquet needs
`pyarrow` (not allowed). The opt-in HF downloader will use plain HTTPS with no
`huggingface_hub`, and will fetch only `.txt`/`.jsonl`/`.json` files and
`.safetensors`/`.gguf` weights. Parquet support would need `pyarrow`. ❓

### 0.10 Smaller notes
- **RTX 3060 Ti has 8 GB** of VRAM and is Ampere (sm_86): bf16 works, FP8/FP4
  don't. It's good for tokenizer work, tiny runs, and as the "modest consumer
  GPU" inference target.
- **PyTorch on GB10 (aarch64, sm_121)** needs a recent CUDA build that ships
  Blackwell kernels. NVIDIA's PyTorch container (NGC) for DGX Spark–class
  machines is the low-friction path. The probe prints `compiled_arch_list` so
  we can confirm the build actually has kernels for your GPU.
- **SwiGLU neurons don't "fire" like ReLU neurons.** SiLU is smooth, so "firing
  frequency" needs a definition (e.g. |activation| above a per-layer threshold).
  We'll report full distributions, not just a count, and make the threshold
  explicit.
- **Checkpoint spacing for the analyzer should be log-spaced** (dense early,
  sparse late), because most specialization emerges early in training.
- **Low precision pays off only for big matmuls.** FP8 speeds up matmuls with
  hidden size ≥ ~1–2k. On a 10M model, the casting overhead wins. FP8/FP4
  *inference* (smaller weights → fewer bytes per token) helps everywhere,
  because of §0.1.

---

## 1. Principles

1. **Local by default.** Exactly one module (`sleuth/net.py`) is allowed to
   open network connections. Every outbound request goes through it and is
   checked against the network policy (§7). A test enforces that no other
   module imports `socket`, `urllib`, or `http.client`.
2. **Dependencies:** Python ≥ 3.11, PyTorch, NumPy. Dev: pytest. The UI server is
   stdlib `http.server`, and the UI is vanilla JS + `<canvas>`, with no npm and
   no build step. Anything else gets asked about first.
3. **Config-driven.** Every model, run, and experiment is a TOML file
   (`tomllib` is stdlib). Each run saves its fully resolved config, git commit,
   and hardware-probe reference next to its outputs.
4. **Readable over clever.** Comments explain *why*. One concept per file
   where reasonable. No metaprogramming.
5. **Everything testable on CPU.** Every component has a tiny config that runs
   on CPU in seconds. GPU-only tests are marked `@pytest.mark.gpu`.
6. **Own formats, open formats.** Model weights are saved as `.safetensors`
   written by our own writer. It's a simple documented format, we need the
   reader anyway, and it's mmap-friendly and pickle-free. Optimizer state uses
   `torch.save` (resume-only, never shared).

---

## 2. Architecture overview

```
               local text / jsonl  ─┐        (opt-in) HF download ─┐
                                    ▼                              ▼
  FORGE   ingest ──► BPE tokenizer train ──► tokenize ──► .bin shards (memmap)
                                                              │
          model config (TOML) ──► Transformer ◄───────────────┘
                                      │  train loop (bf16, accum, cosine, DDP)
                                      ▼
                        runs/<id>/  ckpt/step_000123/{model.safetensors, ...}
                                      │                metrics.jsonl
              ┌───────────────────────┼─────────────────────┐
              ▼                       ▼                     ▼
  INFER   generate (KV cache)   ANALYZER                 LAB
          tok/s, sampling,      taps/hooks ─► stats,     experiment = base config
          spec. decoding        logit lens, ablation,    + overrides × seeds,
                                routing, SAE, timeline   equal-compute compare
                                      │                     │
                                      ▼                     ▼
                            analysis/*.json, *.npy     lab/<exp>/results.jsonl
                                      └──────────┬──────────┘
                                                 ▼
                         UI (localhost dashboard) ◄──► ASSISTANT (optional)
                                                       OpenAI-compatible
                                                       (OpenAI / llama.cpp / vLLM)
                                                       or Anthropic API
  External models: .safetensors / .gguf ──► our readers ──► adapter ──► same Transformer
```

The key design choice is that **external models are mapped onto our own
`Transformer` class** (via adapters) rather than given their own code path.
Then every analyzer feature works on both, for free.

---

## 3. Directory layout

```
Sleuth/
├── PLAN.md  LOG.md  README.md  CLAUDE.md  pyproject.toml  .gitignore
├── configs/          one TOML per run: [model] [train] [data]  (smoke-cpu.toml, tiny-10m.toml, ...)
│   └── lab/          (M6) experiment files: base config + variants + seeds
├── docs/             DATA.md (data sources + licences), images
├── suites/           test suites: <suite>/<category>.jsonl  {"prompt", "expected"?}
├── hardware/         probe results: <hostname>.json  (committed, cited by decisions)
├── scripts/          probe_hardware.py, launch_two_node.sh, ...
├── sleuth/
│   ├── cli.py        `python -m sleuth <command>` entry point (argparse)
│   ├── config.py     TOML → dataclasses, overrides, resolved-config snapshot
│   ├── net.py        THE only network gate (policy, allowlist, logging)
│   ├── paths.py      data/run roots (env: SLEUTH_DATA, SLEUTH_RUNS → NVMe)
│   ├── forge/
│   │   ├── tokenizer/  pretokenize.py, train_bpe.py, bpe.py (encode/decode), special.py
│   │   ├── data/       ingest.py (txt/jsonl walk), shard.py (writer), loader.py (memmap sampler)
│   │   ├── model/      config.py, norm.py, rope.py, attention.py, mlp.py, moe.py,
│   │   │               engram.py, mtp.py, block.py, transformer.py, taps.py, flops.py
│   │   ├── train/      trainer.py, optim.py, schedule.py, checkpoint.py, distributed.py, metrics.py
│   │   ├── infer/      kv_cache.py, sampling.py, generate.py, speculative.py
│   │   └── sft/        chat format, conversation dataset, loss masking
│   ├── formats/      safetensors.py (read+write), gguf.py, gguf_quants.py, hf_tokenizer.py
│   ├── adapters/     llama.py (Llama/Mistral/Qwen), qwen_moe.py, ...
│   ├── analyzer/
│   │   ├── capture.py      tap/hook manager, streaming stats (Welford)
│   │   ├── suites.py       load test suites
│   │   ├── neurons.py      firing freq, magnitudes, dead neurons
│   │   ├── attention.py    head patterns, entropy, induction/previous-token scores
│   │   ├── routing.py      MoE expert counts per category
│   │   ├── ablation.py     zero/mean ablation, deltas
│   │   ├── logit_lens.py
│   │   ├── timeline.py     run any analysis across checkpoints
│   │   └── sae/            activation store, SAE training, feature browser data
│   ├── lab/
│   │   ├── experiment.py   matrix of (variant × seed), budget in FLOPs or tokens
│   │   ├── evals.py        bits-per-byte, suites, synthetic tasks
│   │   ├── synthetic.py    induction, recall, arithmetic task generators
│   │   ├── compare.py      mean ± std, noise floor, significance flags
│   │   └── memory_tiers/   gpu.py, host.py, nvme.py (mmap + async prefetch)
│   ├── assistant/
│   │   ├── backends.py     openai_compat (OpenAI, llama.cpp, vLLM), anthropic
│   │   └── context.py      feeds run/analysis summaries to the model
│   └── ui/
│       ├── server.py       stdlib ThreadingHTTPServer, JSON API, binds 127.0.0.1
│       └── static/         index.html, app.js, charts.js (canvas), style.css
├── tests/            mirrors sleuth/; tiny configs; @pytest.mark.gpu for GPU-only
├── data/             (git-ignored) raw/, tokenizers/, shards/
└── runs/             (git-ignored) <run_id>/{config.json, metrics.jsonl, ckpt/, analysis/}
```

`data/` and `runs/` default to the repo but should point at NVMe via
`SLEUTH_DATA` / `SLEUTH_RUNS` ❓ (which mount points?).

---

## 4. FORGE design

### 4.1 Tokenizer (byte-level BPE)
- **Pre-tokenization:** regex split (GPT-4-style rules, rewritten for stdlib
  `re`), so merges never cross word/number/whitespace boundaries.
  Digits are split into groups of ≤3, which helps arithmetic.
- **Training:** count unique chunks (multiprocessing over files) → BPE on the
  frequency table with an incremental pair-count index and a heap.
- **Special tokens:** `<|bos|>`, `<|eos|>`, `<|pad|>`, and chat:
  `<|system|>`, `<|user|>`, `<|assistant|>`, `<|end|>`. They're reserved IDs,
  never produced by merges, and never matched inside ordinary text unless
  explicitly allowed (this prevents prompt-injection-style token smuggling).
- **Format:** `tokenizer.json` (our own schema: version, regex, merges as ID
  pairs, special tokens). Human-readable and diffable.
- **Tests:** round-trip `decode(encode(x)) == x` on random bytes and unicode,
  special-token handling, determinism, and agreement with a slow reference
  implementation.

### 4.2 Data pipeline
- `ingest`: walk a directory of `.txt` / `.jsonl` (configurable text field),
  normalize newlines, optional exact-dedup by hash, and hold out a validation
  split by document hash, so it's stable when files are added.
- `shard`: write documents as token streams with an `eos` separator into
  `~1 GB` `.bin` files (`uint16` if vocab < 65,536, else `uint32`), plus an
  index of document boundaries (`.idx.npy`) so we can do document-masked
  attention later.
- `loader`: `np.memmap`, random window sampling, deterministic per
  (seed, step, rank), so a resumed run sees exactly the data it would have seen.

### 4.3 Model
A config-driven decoder-only Transformer. Each block is
`x + Attn(Norm(x))` then `x + FFN(Norm(x))` (pre-norm). The pieces are
swappable via config:

| Component | Options |
|---|---|
| norm | RMSNorm (default), LayerNorm |
| position | RoPE (configurable base), none |
| attention | MHA / GQA / MQA via `n_kv_heads`; optional QK-norm; `F.scaled_dot_product_attention` |
| FFN | SwiGLU (default), GELU MLP, **MoE** (top-k router, shared expert option, aux-loss or bias-based balancing) |
| extra | **Engram** module at chosen layers; **MTP** heads; weight tying; **looped** block groups |

`taps.py` gives every interesting tensor a *name*
(`blocks.3.mlp.act`, `blocks.3.attn.pattern`, `blocks.3.moe.router_probs`…).
Taps cost nothing unless the analyzer turns them on. This is cleaner and
faster than generic forward hooks, and the adapters make external models
expose the same names.

`flops.py` counts FLOPs analytically from the config (6N plus the attention
term). The LAB budget and MFU reporting depend on it.

### 4.4 Training
- bf16 autocast (fp32 master weights and optimizer state), gradient
  accumulation, grad clipping, AdamW with no decay on norms/biases/embeddings,
  cosine schedule with warmup (+ optional WSD schedule, which is better for
  checkpoint studies, see §8 ❓).
- `torch.compile` optional (on by default on GPU once verified on each machine).
- **Checkpoints, two kinds:**
  - *Snapshot* (for the analyzer): `model.safetensors` in bf16, plus config.
    Log-spaced + every N steps. Small and cheap; keep all of them.
  - *Resume* (full state): weights, optimizer, scheduler, RNG states, data
    position. Every M steps; keep last K.
- **Logging:** `metrics.jsonl` (step, loss, lr, grad-norm, tokens/s, MFU,
  memory), `events.jsonl`. Plain files that the UI reads.
- **Distributed:** DDP via `torchrun` with NCCL. Single-GPU path is the same
  code with world size 1. For 2 nodes: `torchrun --nnodes 2 --node_rank {0,1}
  --rdzv-endpoint <head-node-ip>:29500`, with `NCCL_SOCKET_IFNAME` set to the
  ConnectX interface (the probe lists interfaces and speeds ❓). Even 1B models
  fit comfortably in 128 GB, so FSDP isn't needed.

### 4.5 Inference
- Own `generate()`: prefill, then incremental decode with a preallocated KV
  cache (GQA-aware). Temperature, top-k, top-p, min-p, repetition penalty,
  stop tokens, seed. Reports prefill and decode tok/s separately (they're
  bounded by different things: compute vs bandwidth).
- Later: self-speculative decoding with MTP heads; KV-cache quantization.

### 4.5.1 Inference speed backlog (from the first GPU run)
At batch size 1 a ~10M model decoded at only ~97 tok/s on the RTX 5060 Ti: overhead-bound,
not bandwidth-bound (see LOG 2026-10-23). In order: CUDA graphs for the decode step, a
static (non-growing) context buffer, sampling on the GPU without a per-token sync, then
batching several prompts. Each one is measured against the previous.

Big quantized models (Gemma 4 12B, Q4_0 kept in VRAM) have the opposite problem: every
token dequantizes all 12B weights in plain PyTorch, so decoding is 0.7 tok/s against
llama.cpp's 46 on the same card. Dequantizing is bandwidth: the lean path (2026-10-24) cuts
the traffic ~4×, but the real fix is a *fused* kernel that reads the 4-bit blocks and
multiplies in one go, never writing a bf16 copy (what llama.cpp does). **Decided
(2026-10-24): Triton** (`triton-windows` on Windows). First kernel: Q4_0/Q8_0
(`forge/model/triton_kernels.py`), checked on CPU in Triton's interpreter; GPU speed to be
measured. Result: Gemma 4 12B decode 0.7 → 7.7 tok/s (llama.cpp: 46). Then a few-row kernel for
batch-1 decoding and a fused RMSNorm. Next: CUDA graphs for the decode step (~4,000 kernel
launches per token are now a large part of the time), then K-quants.

### 4.6 Chat SFT (later)
Conversation jsonl → chat template with special tokens → loss only on
assistant tokens. Reuses the trainer.

---

## 5. ANALYZER design

- **Capture:** enable a set of taps; run a test suite through the model;
  stream **statistics** (Welford running mean/var, max, histogram, threshold
  counts) per neuron/head/layer, broken down by suite category. Only the SAE
  stage stores raw activations (fp16 memmap shards on NVMe).
- **Views** (each is a function returning JSON + `.npy`, saved to
  `runs/<id>/analysis/<step>/<view>/`):
  1. neuron stats: firing frequency, mean/max magnitude, dead/always-on neurons
  2. attention: per-head entropy, attention-to-BOS ("attention sink"),
     previous-token and induction scores, example pattern heatmaps
  3. MoE routing: expert × category counts, load balance, router entropy
  4. ablation: zero/mean-ablate a head/neuron/layer/expert → Δloss and Δaccuracy per category
  5. logit lens: decode each layer's residual through the final norm and unembedding;
     the rank of the correct token per layer
  6. timeline: any view above across all checkpoints of a run
  7. SAE features (later phase)
- **External models** use the same path via `formats/` + `adapters/`.
- **Dashboard:** vanilla JS, heatmaps/timelines/bar charts drawn on `<canvas>`,
  no external assets. Pages: Runs (loss curves) → Checkpoint → View.

---

## 6. LAB design

An **experiment** is a TOML file: a base config, a list of variants
(config overrides), seeds, and a **budget** (training FLOPs or tokens). The
harness:
1. runs (or reuses) the baseline with N seeds to get the noise floor,
2. runs each variant × seed at the same budget,
3. evaluates all with the same battery (val bits-per-byte, synthetic tasks,
   suites, decode tok/s, memory),
4. writes `results.jsonl` and a comparison report (mean ± std; params, active
   params, FLOPs; ⚠️ if within noise).
   Then we add a summary to `LOG.md`, including negative results.

Experiments, in your order, with what I expect and what we measure:

1. **Engram-style conditional memory.** Hashed 2/3-gram IDs → several
   hash-head embedding tables → a gate computed from the hidden state decides
   how much memory to mix in. It goes at a few early/mid layers. Lookups
   depend only on input tokens, so they can be **computed and prefetched
   before the forward pass reaches that layer**.
   - Tiers: GPU tensor → pinned host RAM → NVMe (`mmap` + a thread pool doing
     `pread`, prefetching the next batch's rows while the current batch runs).
   - **Back-of-envelope (to be replaced by probe numbers):** NVMe random reads
     happen in 4 KiB pages, so a 1 KiB row still costs a 4 KiB read. At
     inference (~100 tok/s × ~16 lookups) that's ~2k IOPS, trivial for any NVMe.
     So **SSD-backed tables are very practical for inference**. In *training*
     at ~10⁵ tok/s, that's ~10⁶+ lookups/s, which is at or beyond NVMe limits,
     *unless* a RAM cache absorbs the hot n-grams (n-gram frequency is very
     skewed, so hit rates should be high). We'll measure the hit rate.
   - Optimizer state for giant tables: Adam triples memory. We'll use sparse
     updates with a lighter optimizer for the tables (e.g. row-wise
     Adagrad/SGD). This is itself an experiment.
   - My description of Engram comes from DeepSeek's paper (early 2026) as I
     understand it. If you have a local PDF, put it in `docs/` and I'll check
     the details (hashing, gating, placement) against it ❓.
2. **MoE vs dense** at equal active params, and the **compute (experts) vs
   memory (Engram)** split for a fixed total-param budget.
3. **Looped / recurrent depth:** a shared block group repeated *r* times.
   Compare against unlooped at equal params *and* at equal FLOPs.
4. **Multi-token prediction + self-speculative decoding.** Honest prior: MTP
   tends to help quality only at larger scales, but the speculative-decoding
   speedup is independent of that, and that's what we'll measure.
5. **Low precision:** FP8 training (`torch._scaled_mm`, which is in PyTorch,
   not Transformer Engine), FP4 if the probe confirms kernels, weight-only
   int8/fp8/fp4 inference, and QAT.
6. **My proposals** (each with a reason):
   - **μP (maximal update parametrization):** tune hyperparameters on the
     10M model and transfer them to 350M unchanged. That makes every other
     experiment cheaper and fairer, which is why I'd do it *early*.
   - **Muon optimizer** for hidden matrices: reported large token-efficiency
     gains in small-model speedrun communities, and pure PyTorch.
   - **Distillation from a larger local model** (you already have weights,
     and we're building the loaders): likely the single biggest
     "act bigger" lever. It needs a shared tokenizer, so the student would use
     the teacher's tokenizer via our `tokenizer.json` reader.
   - **Data quality** (near-duplicate removal with MinHash in stdlib, simple
     quality filters): often worth more than architecture changes.
   - **QK-norm + logit soft-capping:** stability, especially under low precision.
   - **Document-masked attention** and sequence packing: free quality at small scale.

---

## 7. ASSISTANT and network policy

- **Backends** (plain `urllib` via `net.py`, no SDKs):
  - *OpenAI-compatible* `/v1/chat/completions`: covers **OpenAI, llama.cpp
    (`llama-server`), and vLLM**, since all three speak this API.
  - *Anthropic* `/v1/messages`.
  - Later: **our own models** as a backend, once SFT exists.
- **What it does:** explain a chart or result (we send it the analysis JSON,
  the run config, and relevant `LOG.md` entries), answer "what does this
  component do", suggest next experiments. It never runs commands or edits
  files on its own.
- **Network policy** in `~/.config/sleuth/settings.toml` (or the repo-local
  `sleuth.toml`):
  - `mode = "offline"` (**default**): only loopback and hosts you list as
    local (e.g. your LAN's private subnet, for a llama.cpp/vLLM server at home).
  - `mode = "online"`: additionally allows the listed internet hosts
    (`api.openai.com`, `api.anthropic.com`, `huggingface.co`).
  - Every request is logged to `runs/net.log`. Downloads always show size
    and ask for confirmation.
- **API keys** come from environment variables only and are never written to
  disk or logs.

---

## 8. Milestones

Each milestone ends with: tests passing, a `LOG.md` entry, and a stop for your review.

| # | Milestone | Exit criteria | Where it runs |
|---|---|---|---|
| **M0** ✅ | Skeleton, PLAN, hardware probe | probe run on both GX10s, the 5060 Ti and the 3060 Ti (results stay local; summaries in LOG.md) | your machines |
| **M1** ✅ (CPU) | Tokenizer + 10M model end-to-end | BPE trains on your data; round-trip tests; 10M model trains on 1 GPU; snapshot + resume checkpoints; `generate` prints text + tok/s; **one analyzer view (per-layer activation stats over checkpoints) in the dashboard** | code + CPU tests here; real run on 5060 Ti or GB10 |
| M2a 🔶 | Scaling ladder (single GPU) | general mix (FineWeb-Edu, Cosmopedia, Wikipedia, Python-Edu, FineMath, sports; Parquet ingest with per-source caps), 32k tokenizer, `mix-20m` / `mix-60m` / `mix-100m` on the same data, `suites/knowledge` + `score` to compare them. *Done: mix-20m (3.34 nats on the whole mix), mix-60m (2.99); next: mix-100m.* | 5060 Ti |
| M2 | Scale-up plumbing | streaming sharding of GBs; val bits-per-byte; DDP 1-node, then 2-node; MFU logging; 125M run | GB10 ×2 |
| **M3** ✅ (CPU) | Analyzer core + dashboard | attention, logit lens, ablation, timeline views; suites by category | any GPU |
| M4 | Assistant + net policy | `net.py` gate + test; OpenAI-compat + Anthropic backends; "explain this" in UI | anywhere |
| **M5** 🔶 | External models | safetensors read/write, GGUF + common quants, `tokenizer.json` reader, Llama-family adapter; analyze a local model. *Done: HF + GGUF loading for llama/qwen2/qwen3 (verified on real Qwen3-0.6B files), export to HF format, `inspect`/`compare`; Gemma 4 GGUF adapter + its tokenizer, quantized-in-VRAM layers (perplexity within 0.07 nats/token of llama.cpp on the owner's 12B file); chat formats (Gemma 4, ChatML) and chat-wrapped analysis; `perplexity`/`next` tools. Fused Q4_0/Q8_0 Triton kernel (first GPU measurement pending). Next: Qwen3's thinking switch, Gemma's MTP draft model (→ M9 speculative decoding), mmproj.* | any GPU |
| M6 | LAB harness | experiment runner, noise floor, compare report, synthetic tasks, μP | GB10 |
| M7 | Engram | module + GPU tier → host tier → NVMe tier with prefetch; quality-vs-table-size and throughput-per-tier results | 5060 Ti + GB10 |
| M8 | MoE | MoE layers, routing analysis, MoE vs dense vs Engram | GB10 |
| M9 | Looped depth; MTP + speculative decoding | results in LOG.md | GB10 |
| M10 | Low precision | FP8/FP4 training & inference, QAT | 5060 Ti + GB10 |
| M11 | Chat SFT; SAEs | chat model; SAE feature browser | GB10 |

M3–M5 are mostly independent, so they can be reordered to match what you're
most curious about.

### About this cloud environment
This Claude Code session runs in a cloud container with **4 CPUs, 16 GB RAM, no
GPU, and no PyTorch installed**. So:
- I can write all the code here and run the CPU tests, **if you allow me to
  `pip install` CPU-only PyTorch and pytest into this container** ❓. (You
  asked me not to install packages without asking.)
- A 10M model can do a few hundred CPU steps here as a smoke test, but the
  real M1 run (and every GPU/probe number) has to happen on your machines.
- **When to pull:** now, to run the probe (instructions in `README.md`), and
  again at the end of M1, to run training on a GPU.

---

## 9. Conventions
- Run IDs: `YYYYMMDD-HHMMSS-<short-name>`; each run directory is self-contained.
- Every decision that depends on hardware cites `hardware/<host>.json`.
- `LOG.md` entries: date, what, why, result (with numbers), surprises.
- Tests: `pytest -q` runs CPU tests in < 1 min; `pytest -m gpu` runs GPU tests.

---

## 10. Open questions for you

1. ~~Packages in this cloud container~~ **Answered: yes.**
2. **Python / PyTorch on your machines:** which versions are installed on
   the two GX10s, the 5060 Ti box, and the 3060 Ti box? Are you using NVIDIA's
   container on the GX10s or a bare-metal install?
3. ~~Data~~ **Answered: none yet.** See `docs/DATA.md`. **New question:** OK to use
   TinyStories next, and is `pyarrow` OK later (for FineWeb-Edu Parquet files)?
4. **Benchmarks:** do you have local copies of any standard eval sets? If not,
   are synthetic tasks + your own suites OK as the evaluation backbone?
5. **External models:** which local models do you want to analyze first
   (architecture + GGUF quant types)? That decides which adapters and dequant
   kernels come first.
6. **Storage:** NVMe mount points on each machine for `SLEUTH_DATA` / `SLEUTH_RUNS`,
   and is the NVMe shared or per node? (For 2-node runs each node needs the
   shards locally, or on a fast shared filesystem.)
7. **Interconnect:** how are the two GX10s connected (direct QSFP cable on
   the ConnectX-7 ports, or a switch)? The probe will show interface speeds.
8. **Parquet:** do you want HF Parquet datasets (needs `pyarrow`), or are
   txt/jsonl only fine?
9. **Engram paper:** do you have the PDF locally to put in `docs/`?
10. **Licence** for the repo (MIT? none/private?).
11. **LR schedule for checkpoint studies:** cosine, as you specified, or also
    WSD (warmup-stable-decay)? WSD lets you branch "decayed" checkpoints from
    any point, which suits the timeline analysis well. My suggestion: support
    both, with cosine as the default.
