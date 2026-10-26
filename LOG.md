# Sleuth: Log

Decisions, experiment results (including negative ones), and surprises.
Newest entries at the bottom. Hardware-dependent claims cite `hardware/<host>.json`.

Format: **date: title.** What, why, result/numbers, surprises.

---

**2026-10-22: Repo skeleton, PLAN.md, hardware probe (M0).**
- Decided: stdlib-only probe (PyTorch optional) so it runs on a bare machine
  before anything else is installed.
- Decided: probe measures *random 4 KiB read IOPS at queue depth 1/4/16/64*,
  not just sequential speed, because the NVMe tier of the Engram tables will
  do small random reads. Sequential GB/s would be misleading there.
- Decided: disk reads use `O_DIRECT` when available (else `posix_fadvise
  DONTNEED`), so we measure the SSD, not the page cache.
- Decided: FP8/FP4 support is reported as *what actually runs*, next to what
  the architecture should support. A PyTorch build without Blackwell kernels
  would otherwise look like missing hardware.
- Decided: model weights will be saved as `.safetensors` written by our own
  writer (we need the reader anyway; mmap-friendly; no pickle).
- Decided: the UI is stdlib `http.server` + vanilla JS + canvas (no npm, no CDN).
- Decided: one module (`sleuth/net.py`) owns all network access; default
  policy is offline.
- Verified in the cloud container (no GPU, no PyTorch): the system/disk/JSON
  paths of the probe work and its tests pass. **GPU sections are untested
  until run on real hardware.**
- Surprise worth remembering: on the GB10, "GPU memory" and "host RAM" are the
  same LPDDR5x pool, so the Engram GPU-vs-RAM tier experiment belongs on the
  RTX 5060 Ti box (see PLAN §0.1).

---

**2026-10-22: Owner's answers.** OK to install CPU PyTorch + pytest in the cloud
container; owner will use a venv on each machine; no training data yet, so I was asked
for guidance plus self-made data that doesn't break any terms of service.
- Cloud container: `download.pytorch.org` is blocked by its network policy, so I
  installed `torch` 2.14.1 (+cu130 build, used on CPU) from PyPI into a venv.
- Researched licences/terms → `docs/DATA.md`. Plan: synthetic now → TinyStories
  (CDLA-Sharing-1.0) for 10M–30M → FineWeb-Edu (ODC-By + Common Crawl ToU; Parquet
  needs `pyarrow`, pending approval) + Wikipedia + Gutenberg for 125M+. Nothing downloaded.
- GB10: PyPI has no sm_121 CUDA build; use the cu130 index or NGC. Community reports
  of attention-kernel failures on sm_121, which the probe's attention-backend check covers.

**2026-10-22: Milestone 1 built (CPU-verified; GPU run pending on owner's machines).**
Decisions:
- *Synthetic corpus* (`synthetic.py`): template stories / arithmetic / tiny Python
  functions with real computed outputs. Three categories on purpose, so the analyzer
  has something to separate. No third-party text.
- *Tokenizer*: GPT-4-style pre-tokenizer rewritten for stdlib `re`
  (`\p{L}` → `[^\W\d_]`, `\p{N}` → `\d`). Merges via incremental pair counts + lazy
  max-heap; a test proves the merges are identical to the naive algorithm. 16 special
  tokens reserved up front (bos/eos/pad, chat roles, 9 spare). Specials are never
  produced from raw text unless explicitly allowed.
- *Shards*: `[bos] doc [eos]`, uint16 when vocab ≤ 65,536, doc-start index saved
  for later document masking. Validation split by content hash (stable when files
  change). Output is byte-identical with 1 or N workers.
- *Loader*: stateless sampling (batch = f(seed, step, micro, rank)), so resume needs
  only the step number and multi-GPU ranks never overlap.
- *Model*: pre-norm, RoPE (rotate-half convention), RMSNorm in fp32, SwiGLU (hidden
  = 8/3·d rounded to 64), GQA through SDPA `enable_gqa`, optional QK-norm, tied
  embeddings, vocab padded to a multiple of 64. Named **taps** at every interesting
  tensor; the same mechanism serves observation and ablation. Explicit attention is
  used only when someone listens to the pattern tap (tests prove it matches the fused path).
- *FLOPs*: PaLM convention (6N + 12·L·d·T); a test checks the analytic param count
  equals the built model's.
- *Checkpoints*: our own safetensors writer; bf16 snapshots (log-spaced) for the
  analyzer, fp32 + optimizer "resume" checkpoints (keep last N). **Resume is
  bit-exact on CPU** (test: 6 steps straight == 4 + resume + 2).
- *Analyzer view 1 (neuron stats)*: a SwiGLU "fire" = |act| > 1× that layer's RMS
  (relative, so comparable across layers/checkpoints; two passes). BOS position
  excluded (attention sink).
- *Dashboard*: stdlib `http.server` on 127.0.0.1 with a strict CSP (`default-src 'self'`),
  vanilla JS + canvas, validated palette, light/dark, mobile-safe. Path components are
  whitelisted (tested against traversal).
- Multiprocessing uses `spawn`, not `fork`. Forking a process that has threads can
  deadlock (pytest flagged it once the dashboard tests ran a server thread). That
  in turn required the `if __name__ == "__main__"` guard in `sleuth/__main__.py`.

Results (cloud CPU, 4 cores, `python -m sleuth demo`, ~0.8M non-embedding params,
300 steps, 0.6M tokens, ~50 s total):
- Tokenizer: 20 MB synthetic → 1,500 distinct chunks; vocab 1024; 3.72 bytes/token.
  Throughput check on 46 MB of Python source: 3.1 s to count (4 workers), 16 s for
  8,000 merges.
- Val loss 2.02 → 0.58 nats/token (0.87 → 0.25 bits/byte). ~16–19k tok/s on CPU.
- Generation: fluent template-style stories, but **names aren't kept consistent**
  ("a little boy named Finn. She… Sam…") and **3 + 4 = 6**. Copying names needs
  induction heads; expect both to improve at 10M on GPU.
- Neuron stats over 16 checkpoints: firing rate drops from 20% at init to ~10% (sparser);
  **category-selective neurons rise from ~5% at init to a peak of ~54% around step 50,
  then fall to ~41% by step 300.** Hypothesis: early neurons latch onto surface features
  that differ by category (code has indentation, math has digits), then become more
  general/shared as the model learns. Caveats: tiny model, short run, a 34-prompt suite.
  The ~5% at init is the measurement's noise floor. Worth re-checking at 10M.

Surprises:
- An untrained model with tied embeddings already favours "repeat the input token"
  (the logit for token x is ≈ ‖E[x]‖²), so a loss test using targets = inputs came out
  *below* ln(vocab). Fixed the test, kept the lesson.

**2026-10-23: First run on the owner's RTX 5060 Ti (Windows, Python 3.12, torch 2.14.1+cu130) found two bugs.**
- **`.gitignore` hid a source package.** `data/` (meant for the top-level data folder)
  also matched `sleuth/forge/data/`, so that package was never committed. Every test
  passed here because the files existed locally; every fresh clone was broken. Fixed by
  anchoring (`/data/`, `/runs/`) and adding `tests/test_repo.py`, which fails if any file
  under the source directories is git-ignored. Verified it catches the original mistake,
  and verified a *fresh clone* passes the tests (the check that would have caught this).
  Lesson: test from a clean checkout, not only the working tree.
- **The probe's disk test was Linux/macOS-only** (`os.readv`, `os.pread`, `O_DIRECT`).
  Replaced with a small `_RawFile` class: O_DIRECT on Linux, `CreateFileW` +
  `FILE_FLAG_NO_BUFFERING` via ctypes on Windows, `F_NOCACHE` on macOS, else plain
  cached reads (labelled as such). Random-read worker errors are now reported instead
  of silently showing 0 IOPS. RAM is now reported on Windows/macOS too.
  **The Windows ctypes path is untested** (no Windows machine here); a test checks byte
  correctness, so running `pytest` on the Windows box will exercise it.
- Windows facts noted: PyTorch on Windows has no FlashAttention (SDPA falls back to
  mem-efficient/cuDNN/math), no Triton (so no `torch.compile`), no NCCL. The CLI now
  writes unencodable characters as replacements instead of crashing on legacy code pages.
- PyTorch 2.14.1 has sm_120 kernels for the 5060 Ti (owner's agent confirmed).

**2026-10-23: RTX 5060 Ti box, first numbers (Windows 11, 32 GB RAM, 24 CPU threads,
PCIe 4.0 platform, NVMe on a separate data drive; torch 2.14.1+cu130, driver 595).**
Hardware probe (full JSON kept locally, not committed):

| | measured | what it means for Sleuth |
|---|---|---|
| fp32 / tf32 matmul | 14.5 / 20.9 TFLOP/s | always train under bf16 autocast |
| **bf16 / fp16 matmul** | **42.0 / 41.4 TFLOP/s** | the MFU baseline for this box (`peak_tflops = 42`) |
| **fp8 (e4m3)** | **81.8 TFLOP/s** (2× bf16) | FP8 is real here; e5m2×e5m2 isn't supported by cuBLAS (normal: e5m2 is paired with e4m3 for gradients) |
| **nvfp4** | **299 TFLOP/s** (7× bf16) | FP4 matmuls *work* in this PyTorch: LAB item 5 can run on this card |
| GPU memory copy | 339 GB/s (read+write; ~76% of the 448 GB/s spec) | VRAM tier of the Engram tables |
| host→GPU (pinned / pageable) | 14.0 / 11.3 GB/s | consistent with the card's x8 link on PCIe 4.0. **24× slower than VRAM**: the RAM tier is a real cliff on this box |
| NVMe sequential read / write | 6.04 / 3.16 GB/s (uncached, FILE_FLAG_NO_BUFFERING) | |
| NVMe random 4 KiB | 13.5k IOPS at QD1 (73 µs), 52k at QD4, ~63k at QD16–64 | see Engram note below |
| attention kernels | efficient ✔, cuDNN ✔, math ✔, flash ✘ (not in Windows builds) | SDPA picks a working kernel automatically |

Engram tier arithmetic for this box (to be measured properly in M7): with ~16 table
lookups per token and one 4 KiB read per lookup, ~60k IOPS from Python threads caps a
pure-SSD table at **~3.8k tokens/s**. That's plenty for inference (~100 tok/s) but ~40×
too slow for training at ~145k tok/s, so training needs the hot rows cached in VRAM/RAM
(a >97% hit rate). 60k IOPS is probably the Python-threads limit, not the drive's
(good NVMe drives do several hundred thousand+). A native async I/O path (Windows
IoRing / Linux io_uring) is a candidate experiment.

First GPU training run: `tiny-10m` on the synthetic corpus (9.84M params, 5000 steps,
164M tokens):
- **~145k tok/s, about 25% MFU** (73.1 MFLOPs/token × 145k = 10.6 TFLOP/s of 42). That's
  reasonable for a 10M model without FlashAttention or torch.compile; small matmuls
  can't fill the GPU.
- **Overfitting, as predicted by the data size:** only 5.3M training tokens, read ~31
  times. Val loss bottomed at ~0.213 around steps 2000–2250 and rose to 0.291 by step
  5000 while train loss kept falling (0.130). The trainer now warns when a run will
  read its data more than ~4 times. The `tinystories-10m` config reads its data
  less than once.
- Generation: complete, coherent template stories that stop at <|eos|>, and **names
  now stay consistent** ("a little boy named Tom… Tom and Finn were best friends"),
  unlike the 1M CPU model. Likely evidence of copying/induction behaviour at 10M,
  to be confirmed in the attention view (M3).
- **Decode is only ~97 tok/s**, although the weights (20 MB) could be read ~20,000×
  per second at this bandwidth. At batch size 1 a tiny model is bound by per-token
  overhead, not memory bandwidth: ~100 small kernel launches per token (slower under
  Windows' driver model), a GPU→CPU sync per token, and the sampling ops. CUDA graphs
  (record the decode step once, replay it) are the standard fix; added to the inference
  backlog. This is a good example of "measure before you optimize": the bandwidth
  bound from PLAN §0.1 only matters for bigger models.

Bugs found by the run, all fixed:
- `test_txt_separator_and_jsonl` failed on Windows (`write_text` turns `\r\n` into
  `\r\r\n`), so the test now uses `write_bytes`. A test bug, not a loader bug.
- The probe crashed writing JSON with `--disk-path` (a *list* of Paths). Fixed, and
  tested with that flag. The probe also no longer records the GPU UUID, and shows the
  home folder as `~`.
- `demo` crashed on Windows: it built a TOML string from a Windows path (`\s` is an
  escape). It now sets the path directly, and `--set` raises a clear error for
  malformed quoted values instead of keeping the quote marks.
- The tok/s spike after each eval (2×): eval_every (250) isn't a multiple of log_every
  (20), and resetting the timer after an eval paired 20 steps of tokens with 10 steps
  of time. The timer now skips the eval's duration instead.
- `generate` printed text after generation and its stats line out of order. It now
  streams token by token (incremental UTF-8 decoding), flushes both streams, and does
  a warm-up forward so the first sample's prefill speed isn't skewed by one-time setup.

Owner decisions: **TinyStories approved** as the next dataset; **`pyarrow` approved**
for reading Parquet (FineWeb-Edu) later. The repo is public, so hardware JSON files
are now git-ignored and personal/machine identifiers were scrubbed from the docs.
At the owner's request the branch history was then rewritten (`git filter-branch`)
so no earlier commit contains the node names or the LAN IP either, and force-pushed.
Commit SHAs changed; the final file tree is byte-identical.

**2026-10-23: Repository reset.** The owner deleted and re-created the GitHub repo so no
history would contain personal details. The re-upload was an older copy that still had the
node names and a LAN IP, so it was replaced with one clean commit holding the latest files
(audited: no host names, usernames, emails, home paths, GPU IDs or LAN IPs). The new repo
wasn't yet visible to the Claude GitHub App (HTTP 403 on push), so the clean commit was
handed over as a git bundle for the owner's machine to push.

**2026-10-23: Milestone 3 built: analyzer core (CPU-tested).**
New views, each run on every checkpoint, each a dashboard tab with timelines + detail:
- *Attention heads*: per-head share of attention to <|bos|> (sink), previous token, self,
  normalized entropy; **induction score** on repeated random tokens ([bos] + R tokens + the
  same R again), plus a model-level *induction gain* (loss on the first copy minus the
  repeat). Pattern viewer for one suite prompt (click a head to open it).
- *Logit lens*: final norm + unembedding applied after every layer; per-layer loss and
  top-1 accuracy by category; a layer × position grid for one prompt. A test checks that
  the last row equals the model's real loss.
- *Ablation*: mean (default) or zero replacement of each head (`attn.z` tap), each
  layer's attention and MLP; Δ loss, Δ expected-answer loss and Δ exact match per
  category. BOS position left untouched. Tests check that the model is unmodified
  afterwards, and that ablating a component whose output never varies changes nothing.
- *Scoring* (`analyzer/common.py`): teacher-forced; prompt and expected answer encoded
  separately so the answer tokens are known exactly.
- New suite `suites/stories` (names / objects / dialogue / narrative) for TinyStories models.
- Timeline format generalized: every view emits named *series* (one value per row: layer,
  head, or "model"), so the dashboard draws any view without view-specific timeline code.

First observations (1M synthetic CPU model, so tiny and out of its depth):
- **No induction yet**: every head scores < 4% induction attention, and the induction
  gain is ≈ 0 (−0.2 at step 300: it's *worse* on the repeat). That's consistent with it
  mixing up names. Expect a clear jump for the 10M TinyStories model.
- Logit lens (stories suite, step 300): top-1 accuracy 0% (emb) → 5% → 12% → 14% → 14%;
  layers 1–2 do most of the work.
- Ablation, a nice lesson: on its own synthetic suite, removing layer 0's MLP costs
  **+2.4 nats**. On the unfamiliar stories suite, removing MLPs *lowers* the loss (blue):
  the model is confidently wrong there, and ablation just reduces its confidence.
  "Ablation helps" on out-of-distribution data usually means overconfidence, not
  that the part is harmful.

**2026-10-24: Milestone 5, part 1: other people's models.**
The owner has a Gemma GGUF (`gemma-4-12B-it-qat-UD-Q4_K_XL.gguf`, plus mmproj and MTP files)
locally, and Qwen/Nemotron/GLM/DeepSeek safetensors on the cluster. Plan: validate on a
small Qwen first (simplest architecture, both formats available), then Gemma.
- *Model changes*: `head_size` (Qwen3's head width ≠ d_model/heads), `qkv_bias` (Qwen2),
  and `Transformer.empty()`, which builds on the meta device so a 12B model isn't randomly
  initialized first (minutes of work and a full extra copy in RAM).
- *Hugging Face*: config.json → ModelConfig, sharded safetensors via the index file, name
  mapping; anything we can't reproduce exactly (rope_scaling, non-SiLU activations)
  raises instead of loading a subtly wrong model. Plus an **exporter** (our model →
  HF format), which is also the route to serving our models with llama.cpp/vLLM.
- *tokenizer.json*: byte-level BPE with our own `\p{L}`/`\p{N}` → `re` translator,
  including negated classes (`[^\s\p{L}\p{N}]` → `(?:(?![^\W_])[^\s])`). Test: our
  tokenizer exported and re-imported gives identical token IDs.
- *GGUF*: own reader (metadata, tensor table, alignment, memory-mapped) and dequantizers
  in vectorized PyTorch (CPU or GPU) for F32/F16/BF16, Q4_0/Q4_1/Q5_0/Q5_1/Q8_0, Q2_K–Q6_K,
  IQ4_NL, IQ4_XS. Tested two ways: reference quantizer → dequantize round-trips for the
  simple formats, and a line-by-line transcription of ggml's C decoders vs. the vectorized
  versions for the K-quants. **Both come from my reading of ggml's code, so a misreading
  would pass both tests.** The real check is a real model (next item).
- *GGUF → our model* for llama/qwen2/qwen3. llama.cpp's converter reorders q/k rows for
  the `llama` arch (interleaved RoPE). Verified numerically that the reordering + interleaved
  RoPE gives the same attention scores as our rotate-half on the original rows, so we undo it.
  Qwen2/3 use NEOX RoPE and aren't reordered.
- *Tools*: `sleuth inspect` (file contents, tensors grouped by pattern), `sleuth compare`
  (two models, same tokenizer: top-1 agreement + KL), and `analyze` on external models
  (results in `runs/ext-<name>/`). The dashboard hides training and timeline panels when
  there's only one checkpoint.
- Logit lens now unembeds one layer at a time: stacking all layers is rows × T × V, i.e.
  gigabytes for a 150k vocab.
- A bug caught by the dashboard check: charts drawn into a hidden panel got zero-size
  canvases, and one zero-size `drawImage` aborted the whole page. Hidden panels are now
  skipped, and charts refuse to draw into anything under 10 px.
- Sanity result on the 1M smoke model: original vs. its bf16 HF export vs. its Q8_0 GGUF
  export (llama arch, with the row reordering) generate **word-for-word the same story**;
  `compare` HF-bf16 vs. GGUF-Q8_0: 99.6% top-1 agreement, mean KL 0.0001 nats.

Proposed validation on real weights: Qwen3-0.6B as official safetensors vs. ggml-org's Q8_0
GGUF vs. a Q4_K_M GGUF, using `sleuth compare`. Same model in three formats, so any
disagreement beyond quantization noise points at our loader or dequantizer.

**2026-10-24: First TinyStories model (RTX 5060 Ti, Windows).** `configs/tinystories-10m.toml`:
12.59M params (9.44M non-embedding), vocab 8192, 6000 steps × 64 × 512 = 197M tokens in ~25 min.
- Data: TinyStoriesV2-GPT4 → 531M train / 2.7M val tokens at 4.14 bytes/token (shards built in 41 s).
  The run used 37% of one epoch, so no data was repeated.
- Speed: 125–137k tok/s, **~26% MFU** against the re-measured bf16 peak (44.75 TFLOP/s).
- Validation, at steps 250 / 500 / 1000 / 2000 / 3000 / 4000 / 5000 / 6000:
  - loss 2.52 / 2.05 / 1.80 / 1.62 / 1.52 / 1.45 / 1.41 / **1.385** nats
  - bits/byte 0.879 / 0.716 / 0.627 / 0.564 / 0.531 / 0.507 / 0.491 / **0.483**
- It **never overfit**: val loss fell at every check, and train loss (~1.37) stayed level
  with val at the end. Still improving at step 6000, so a longer run (or a bigger model)
  would get lower. That's a natural first LAB baseline.
- Samples at 6000: grammatical, with dialogue, stable character names ("Tim … Tim"), story arcs
  and morals. Logic slips remain ("tied the string to the string", "joy and joy").
  At step 256 it's grammatical but loops on a word ("the fish … the fish … a big fish").
  Asked "what does 5 + 10 equal", it tells a story about the park: TinyStories contains
  no arithmetic, so it has never seen any. A good reminder that a model's abilities are bounded by its data.
- Re-run probe (GPU only): bf16 44.75, fp16 42.8, fp8 e4m3 90.0, nvfp4 **334** TFLOP/s;
  memory copy 351 GB/s; host→GPU 14.1 GB/s. That's 3–10% above the first probe (the first was
  probably sharing the GPU with something). The MFU baseline in the config is now 44.75.
- Fixed after this run: `--step` now falls back to the nearest snapshot (log-spaced
  snapshots meant step 250 didn't exist; it uses 256 and says so). The first sample's
  prompt still ran slowly (~30 tok/s vs. 300+ later) despite the warm-up: the warm-up ran
  only a forward pass and left sampling and KV-cache setup cold. It's now a throwaway
  2-token generation. Prompt speed is reported in milliseconds, since "tokens/s" over a
  5-token prompt is mostly noise.
- Still slow, as expected: decode ~90 tok/s at batch 1 (per-token overhead, see the
  inference backlog in PLAN §4.5.1).

**2026-10-24: What the Qwen3 / TinyStories test round showed, and the fixes.**
- *Loader correctness, settled:* safetensors vs ggml-org's Q8_0: KL 0.0075 nats, 96.9% top-1
  agreement, word-for-word the same greedy text. The Q4_K_M per-tensor weight errors are
  at the level 4-bit rounding *should* give: I simulated plain round-to-nearest on random
  data (asymmetric 4-bit, blocks of 32: 7.8% Gaussian, 9.2–9.4% heavy-tailed; symmetric
  6-bit, blocks of 16: 2.1–2.6%), and the file's Q4_K (7.3%) and Q6_K (1.8%) are slightly
  *better*, as llama.cpp's scale search + imatrix should be. A misread bit layout gives
  errors near 100%. So the dequantizers are right.
- *Q4_K_M's large KL (0.66) is probably real, not a bug,* but that's not proven. New
  `compare --requant q4_1` rounds the original weights through our own 4-bit quantizer:
  if that alone gives a similar KL, Qwen3-0.6B is simply that sensitive (small models
  usually are). `--requant-match` narrows it to some weights (e.g. only the embedding).
- *Qwen3-0.6B, analyzed:* induction gain 14 nats/token (random-sequence loss 14.9 → 0.9),
  top head L16H14 at 0.96. The logit lens is near useless until layer ~17 (top-1 < 10%),
  then climbs to 36%: big models' middle layers don't use the "output language" yet
  (the motivation for the tuned lens). Our 10M TinyStories model is the opposite: its top-1
  accuracy climbs from layer 0.
- *TinyStories induction looked weak* (top head 0.09; random-sequence loss only 12.2 →
  11.3), yet the gain jumped between steps 256 and 500 (0.13 → 0.61), a classic induction
  phase change, and the top induction heads L3H1/L3H2 are the most important single heads
  for the *names* category in ablation. Hypothesis: the test drew random tokens from the
  whole vocabulary, which for an 8k TinyStories tokenizer is full of tokens (raw bytes,
  fragments) the model barely saw, so it can't match them. The view now draws from tokens
  that occur in the suite, distinct within a sequence, and keeps the old number as
  "induction gain, any token". To be checked on the owner's run.
- *Ablation:* removing layer 0's MLP costs +5.4 nats on names, far more than any other
  part: in small models the first MLP acts like an extension of the embedding (seen in
  GPT-2 too), so nothing downstream works without it.
- Things that were my mistakes, not bugs: the analysis covered 26 checkpoints because the
  snapshots are 14 log-spaced (0, 1, 2, …, 4096) + 12 every-500 ones, not 19; and the
  test count was 129, not 128.
- Fixed: `inspect` printed only the first tensor's shape per group, which hid that Gemma's
  full-attention layers are bigger (see below); `--show-key` prints one metadata value in
  full. GGUF `output.weight` that is a byte-for-byte copy of `token_embd` is now tied
  (if ggml-org's Qwen3-0.6B Q8_0 copy is exact, it should load as 596M params, not 751.6M:
  −311 MB in bf16; to be confirmed). The dashboard's ablation
  ranking follows the Measure/Category selectors. The logit lens explains its "emb" row
  (with tied embeddings it re-predicts the current token: 105.8 nats for Qwen). External
  runs are named `ext-<parent>-<name>-<path hash>` and refuse to mix two models' results.
- Slow Qwen decode (19 tok/s): per-token overhead, as for the 10M model. A 28-layer model
  launches ~1,500 small GPU kernels per token, at tens of µs each under Windows. CUDA
  graphs stay first in the inference backlog.

**2026-10-24: Gemma 4 adapter (GGUF, dense, text) and quantized-in-VRAM layers.**
Built from the owner's `inspect` output plus llama.cpp's own code for this architecture
(`src/models/gemma4.cpp`, tokenizer in `src/llama-vocab.cpp`, read online, nothing
downloaded into the repo), which is the reference for how GGUF files run.
- *Architecture, from the file:* 48 layers, d = 3840, 16 query heads, GeGLU MLP of 15,360.
  Five *sliding-window* layers (window 1024, head width 256, 8 KV heads, RoPE base 10k)
  per *full* layer (8 of them, the first is layer 5: head width 512, one KV head, RoPE
  base 1M, no `attn_v`). The inspect sizes give this away even though the old inspect showed one
  shape: attn_q's 472.5 MiB only adds up if 8 layers have 8192 rows instead of 4096.
- *Surprises from llama.cpp's graph:* attention scale 1.0 instead of 1/√head_dim (q and k
  are RMS-normalized, so their size is already fixed); on full layers the values *are* the
  keys (before k_norm and RoPE); values get an RMS norm without a gain; every sublayer
  output is normalized again before it's added ("sandwich" norm); after each layer the whole
  residual stream is multiplied by a learned scalar (`layer_output_scale`); embeddings
  × √3840; logits soft-capped at ±30; the full layers' RoPE frequencies are divided by
  factors stored in `rope_freqs.weight`.
- *Our model* gained all of these as config options (off by default), plus a per-layer
  attention spec, one RoPE table per layer kind, and a KV cache with per-layer shapes.
- *Tokenizer:* SentencePiece-style BPE: spaces → "▁", split into lines only (merges can
  cross words), BPE over characters, unknown pieces → `<0xXX>` byte tokens.
- *Quantized in VRAM:* bf16 would be ~24 GB, the 5060 Ti has 16. `QuantLinear` /
  `QuantEmbedding` keep the file's blocks (Q4_0: 6.2 GiB) and dequantize one layer at a
  time; the embedding gathers only the rows it needs, and the tied 262k-row output layer
  dequantizes in chunks. Automatic when bf16 wouldn't fit in half the GPU. Cost: every
  forward pass dequantizes every weight again (CPU: ~0.15 s per 59M-weight matrix), so
  decoding will be slow; fused dequant+matmul kernels are the fix, later.
- *How it's tested:* a tiny random Gemma 4 GGUF with the real file's structure, compared
  against a plain-tensor transcription of llama.cpp's graph (matches to 2e-4, also with
  Q8_0/Q4_0 weights kept quantized), and the tokenizer against a transcription of
  llama.cpp's queue-based BPE. **Same caveat as the K-quants: both come from my reading of
  llama.cpp, so the first real run is the actual check.** Unsupported variants (E2B/E4B
  per-layer embeddings, KV sharing, MoE, partial RoPE) raise an error.
- *Not yet:* the chat template (Jinja; we'll need to see it), the MTP "assistant" file
  (4 layers of width 1024 with no K/V of their own, `shared_kv_layers = 4`: it reads
  the main model's KV cache; `nextn.pre_projection` maps 7,680 = 2 × 3,840 inputs, plausibly
  the main model's hidden state + a token embedding, to 1,024), and the mmproj file
  (`block_count = 0`: only patch/position embeddings and projections, no encoder layers).
  Both belong with speculative decoding (M9) and multimodal input, later.

**2026-10-24: Round 3: Qwen's 4-bit loss is real, the Gemma 4 forward pass matches llama.cpp, chat formats.**
- *Qwen3-0.6B is simply sensitive to 4-bit.* Rounding the original weights through our
  plain Q4_1 quantizer (8.0% weight error) costs KL 0.93 and 69% top-1 agreement:
  *worse* than bartowski's Q4_K_M (KL 0.66, 78%), as it should be, since K-quants with an
  imatrix are smarter. Nearly all of it comes from the block weights (KL 0.86); the
  embedding alone costs 0.08. Q8_0 rounding: KL 0.009. So the K-quant decoders are
  confirmed end to end, and "0.6B models hate 4-bit" goes in the notebook. The tied
  output fix works: Qwen3-0.6B Q8_0 loads as 596M params.
- *Gemma 4 forward pass, checked against llama.cpp by the owner.* Same tokenization, and
  `llama-perplexity -c 256` on TinyStories text: llama.cpp 22.59, ours 20.98 (0.07 nats/token
  apart; per position within 0.3–0.4 nats). Not identical, and not expected to be: llama.cpp
  rounds activations to 8 bits for its quantized matmuls and keeps the KV cache in f16; we
  run bf16 activations. A run of ours in fp32 will show how much is our own rounding.
  Greedy text is the same degenerate "111…" in both on raw prompts.
- *Gemma 4 IT on raw text is bad at the start of a sequence, in both implementations:*
  ~19 nats/token at positions 1–7, ~3 by positions 128–255. Inside its chat format it
  answers properly (llama.cpp: "Paris"). So the analyzer suites, which are short raw
  prompts, measured nothing useful for Gemma (logit-lens final loss 12.0 vs uniform 12.5).
  New: chat formats written out from the GGUF's template (formats/chat.py; Gemma 4 and
  ChatML), `generate/next --chat`, and `analyze/compare --chat`: each test becomes the
  model's reply to a user turn ("Continue this text."), only the test's own tokens are
  scored, and results go to a separate "<suite>-chat" suite.
- *Gemma 4 attention (raw prompts, so provisional):* induction gain 14.0 nats/token on
  random suite tokens (18.1 → 4.1); top induction heads L41H14 (0.67), L17H0 (0.59).
- *TinyStories induction: my hypothesis was wrong.* Drawing the random tokens from the
  suite's own 258 tokens instead of the whole vocabulary changed almost nothing (gain 1.04
  vs 1.03 at step 6000, top head still L3H2 at 0.08). The model is confidently wrong on
  random sequences (first copy 11.5 nats, above uniform-over-the-vocabulary 9.0) and the
  repeat only helps a little. New measure: each suite text fed twice ("repeated text"),
  i.e. copying in the setting the model was trained on, with per-head scores.
- *Bug:* `train()` switched fp32 matmuls to TF32 for the whole process and never switched
  back, so a later exactness test in the same pytest run failed (diff 0.03). It now
  restores the setting. Found by the owner's agent.
- *Gemma's "emb" row top-1 is the current token in 0% of positions* (Qwen and TinyStories:
  100%). Tied embeddings make self-prediction likely, not certain: Gemma's final norm has
  per-channel gains that reshape the vector first (my guess at the reason). The dashboard
  note no longer claims it always happens.
- *Speed:* Gemma decodes at 0.7 tok/s vs llama.cpp's 46. Every token dequantized all 12B
  weights through int64 temporaries (~50 bytes of memory traffic per weight). New lean
  Q4_0/Q8_0 path: unpack straight into bf16 and scale in one pass, same values exactly;
  2.3× faster on the CPU, GPU still to be measured. The logit lens now unembeds many layers
  per call (it was dequantizing Gemma's 1B-weight output layer 49 times per test). Matching
  llama.cpp needs a fused dequantize-and-multiply GPU kernel: an owner decision (Triton or a
  small CUDA extension), see PLAN §4.5.1.
- New tools: `perplexity` (llama-perplexity's chunking, plus bits/byte and loss by
  position) and `next` (top next-token candidates with probabilities).

**2026-10-24: Triton kernel for quantized models; data for a scaling ladder.**
- *Owner decisions:* Triton for fused GPU kernels (`triton-windows` 3.8.x on Windows, matching
  PyTorch 2.14; Linux PyTorch ships Triton). And: build a model ladder bigger than 10M on
  general knowledge, Python, math and sports, to see how answers improve with size.
- *Fused Q4_0/Q8_0 matmul* (`forge/model/triton_kernels.py`): each program loads a slice of
  activations and the matching raw weight bytes, unpacks and scales them in registers
  (fp32, rounded once to bf16, the same values as the plain path), and accumulates with
  `tl.dot`. Each weight byte is read once and nothing is written back, against ~30 GB of
  memory traffic per token before. Checked on CPU with Triton's interpreter
  (TRITON_INTERPRET=1): matches dequantize+matmul to fp32 rounding, also through a whole tiny
  Gemma 4. The interpreter has no bfloat16, so bf16 is checked on the GPU only: a
  self-check on first use compares the kernel with the plain path and silently falls back
  if they disagree (or Triton fails to compile). `SLEUTH_NO_TRITON=1` forces the plain path.
- *Data:* Parquet ingest (pyarrow, approved earlier) and per-source caps `path@MB`, which set
  a mix's proportions for both the tokenizer and the shards; `meta.json` records the actual
  mix. Sources and licences in docs/DATA.md §2a: FineWeb-Edu 40%, Cosmopedia v2 15%,
  Wikipedia 12%, Python-Edu 15%, FineMath-4+ 10%, OnlySports 8% (optional: CC BY-NC-SA).
  One surprise from the research: SmolLM's own `python-edu` subset holds only file IDs (the
  code must be fetched from Software Heritage's S3), so we use a community copy with the
  code included (`Avelina/python-edu-cleaned`, ODC-By).
- *Ladder:* `mix-20m` (22M params, 0.5B tokens, ~2 h estimated), `mix-60m` (64M, 1.0B, ~9 h),
  `mix-100m` (101M, 1.57B, ~20 h): same 32k tokenizer, data and 1024-token context, so only
  size differs. New `suites/knowledge` (facts, sports, math, python; 40 short questions) and
  a `score` command (loss, answer loss, exact match per category) to compare them.

**2026-10-25: Round 4: Triton works on Windows; first scaling ladder (20M, 60M); a val-loss bug.**
- *Triton on the RTX 5060 Ti* (triton-windows 3.8.0.post29 + PyTorch 2.14.1): all GPU tests pass,
  including the bf16 kernel check and torch.compile. Gemma 4 12B decoding went 0.7 → **7.7 tok/s**,
  prompt 1593 → 127 ms (the plain path, with the lean dequantizer, is 1.6 tok/s). llama.cpp: 46.
  What's left at batch size 1: the dot-product kernel wasted 15 of its 16 tensor-core rows and
  reloaded each block's scale per weight (new few-row kernel, CPU-checked; GPU speed to be
  measured), and ~4,000 small kernel launches per token (RMSNorm is now one fused op; CUDA graphs
  are the real fix, next).
- *Gemma 4 forward pass, settled:* perplexity 22.72 (bf16) / 22.46 (fp32) vs llama.cpp's 22.59,
  under 1% apart. Last round's 20.98 came from a scratch script with slightly different chunking.
  Raw-prompt first token: ours "0" (31%) vs llama.cpp "1"; our margin is 0.375 logits, two bf16
  steps at that size, so plausibly rounding; an fp32 `next` will tell.
- *Gemma in its chat format* answers "Paris" (100%) and scores 77.5% exact on suites/knowledge
  (answer loss 0.32). Chat-wrapped logit lens: final loss 4.08, top-1 42% (raw prompts: 12.0 /
  12%). Its last layer does a lot: L46 is at 12.4 nats, L47 at 4.1. Top induction heads agree
  across both measures: L17H0, L41H14, L35H7.
- *Induction depends on the data.* The mix models grow strong induction heads early: random-token
  gain 1.6 (20m) / 5.5 (60m) at step 500, ~10.3 by the end (first copy 11.2 → repeat 0.8), with
  heads scoring 0.75–0.94 (L5H0/L5H2 in 20m; L4H1/L4H2/L4H9 in 60m). The TinyStories model never
  did (top head 0.12, gain 1.0), and its gain on repeated *story text* peaked at step 500 (0.45)
  then sagged (0.37). TinyStories rarely repeats long exact strings; web text and code do all the
  time, so copying pays off there. Bigger model → earlier and sharper induction heads.
- *Scaling, first two rungs* (same 32k tokenizer and data; val = 40 random windows of the whole
  mix): mix-20m (22M params, 0.5B tokens) 3.34 nats → mix-60m (64M, 1.0B) **2.99**. No overfitting
  (val = train). On suites/knowledge, answer loss 3.16 → 2.79 (sports 3.88 → 3.24, python 2.02 →
  1.65), but exact match stayed at 7.5% (the same 3 Python items): knowledge is moving under the
  surface before it shows up as right answers. Greedy text: fluent nonsense ("The capital of France
  is France"), and Q&A prompts slide into multiple-choice formats copied from the training data.
  Logit lens final row: 3.18 / 40% top-1 (20m) → 2.76 / 49% (60m).
- *Training speed:* micro-batch 16 at 64M filled the 16 GB and Windows spilled into shared memory
  (17% MFU). Micro-batch 8 + torch.compile: **~63% MFU, 63k tok/s, ~8 GB** (owner's agent's
  benchmark: compile alone is ~2.5×). The ladder configs now use that, with `eval_windows = 640`.
- *Bug (found by the owner's agent): the validation loss only measured FineWeb-Edu.* Eval took
  windows from the start of the val split, and shards store documents source by source, so with
  40 batches it read the first 7% = FineWeb-Edu only (3.90 reported vs 3.34 over the whole mix for
  mix-20m). Now: windows spread evenly over the whole split, and shards record where each source
  starts, so every eval also reports loss *per source*. `sleuth eval RUN` re-measures a finished run
  (for the per-source split, re-shard: ~2 min). TinyStories numbers were unaffected (one source).
- *Other fixes:* a STOP file (`runs/<run>/STOP`) and SIGTERM/Ctrl-Break also stop training with a
  checkpoint (a Ctrl-C sent to a background process never arrived, losing ~460 steps); peak memory
  in the training log; meta.json no longer records an absolute local path (the shard folder keeps a
  copy of its tokenizer); `tokenizer` prints bytes/token; compile is skipped with a warning if
  Triton is missing; `score --few-shot` (worked examples first, so a base model knows to answer
  briefly).

**2026-10-25: After the round-4 review: a bigger knowledge suite, loop-free samples, stale runs, and a dashboard that runs jobs (D1).**
The owner's round-4 review (all 181 CPU + 6 GPU tests passing on Windows) suggested an order:
first the knowledge suite and repetition controls, then dashboard Jobs + Playground. Both are
done here, plus bug 2. PLAN §8.1 now holds the roadmap.
- *`suites/knowledge`: 40 → 220 items* (70 facts, 50 sports, 50 math, 50 python). The first 10 of
  each file are the old items with the same IDs. With 40 items one answer moved exact match by
  2.5 points and every model got the same 3 Python items right. `score` now prints **answer loss
  first** (it's the signal at these sizes: 3.16 → 2.79 from mix-20m to mix-60m while exact match
  sat still) with **± one standard error** per category, and `--show-items` / `--json` show
  each item and the model's top guess. A test runs every Python "# prints" item and checks the
  expected output, so a wrong answer key can't quietly make every model look worse. Old
  whole-suite numbers aren't comparable with the new suite; re-score the ladder once.
  Note: exact match at 0% shows "± 0.0%" (the formula's SE is 0 when p = 0). With 220 items the
  honest reading is "under ~1.4%".
- *Sampling:* `--repetition-penalty` already existed, but looked at the whole context. Now it
  looks at the last `--repeat-last-n` tokens (64, as in llama.cpp), and `--no-repeat-ngram N` bans
  any token that would repeat an N-token phrase (Hugging Face's no_repeat_ngram_size; incremental,
  O(1) per step). It's a hard ban, so it also works greedily: the tiny test model's greedy loop
  disappears with N = 3. Scoring never samples, so none of this touches `score`/`analyze`.
- *Bug 2, stale `runs/ext-hf`:* a run is stale when it's an external-model folder whose name
  doesn't end in its model path's hash and whose config lacks `keep_quantized` (added in the same
  round as the hash; a hand-picked `--name` from a current Sleuth has it). If another run holds
  the same model, it says which. The dashboard hides stale runs (a "stale" checkbox shows them);
  `sleuth runs` lists them, `sleuth runs --prune` shows what it would delete, `--prune --yes`
  deletes. Nothing is deleted automatically.
- *Dashboard Jobs (D1).* Decisions:
  - **Allow-list, not a command box.** Each command (train, resume, eval, score, analyze,
    generate, next, tokenizer, shard) has typed fields; the server validates every value and
    builds the argument list itself (`subprocess.Popen([python, -m, sleuth, ...])`, no shell).
    Options are passed as `--flag=value`, so a prompt starting with "-" can't turn into an option;
    positional values starting with "-" are refused. The page builds its forms from the same
    list (`/api/jobs/specs`).
  - **Two lanes, one GPU.** GPU jobs run one at a time, the rest queue; tokenizer/shard (and
    anything with device=cpu) use a CPU lane alongside. The Playground takes the GPU lane while
    it has a model on the GPU, so a training run and Gemma's 7.7 GB can't collide.
  - **Stopping training uses the STOP file**, but only once the trainer has printed "to stop
    cleanly…": it deletes any old STOP file when it starts, so earlier we send SIGTERM / Ctrl-Break.
    Why not always a signal: it can land in the middle of `optimizer.step()`, and the checkpoint
    saved then would hold half-updated weights. The STOP file is checked between steps. Second
    click = kill.
  - **Jobs outlive the dashboard**: each child gets its own session/process group. Surprise worth
    noting: on Linux, Ctrl-C in the terminal goes to the whole foreground process group, so without
    this, closing the dashboard would have interrupted every training job it started. After a
    restart, running jobs show as "orphaned" (the log keeps growing and is still shown).
  - **Logs** go to `runs/_jobs/<id>.log` and stream to the page with Server-Sent Events (plain
    HTTP; whole lines only, so a multi-byte character is never cut; resumes from the last offset
    on reconnect). Job history is one JSON file per job in the same folder.
  - **Security, now that the page can start processes:** a random per-session token (printed in
    `sleuth ui`'s link, kept by the page, sent as `X-Sleuth-Token`) on every POST; POSTs from
    another Origin are refused; and every request must name the server as 127.0.0.1/localhost in
    its Host header. The last one is DNS-rebinding protection: an attacker's domain can resolve
    to 127.0.0.1, which would let their page *read* our API as same-origin; their requests still
    carry their own host name.
- *Live curves:* a run counts as live if a job is working on it or its metrics.jsonl changed in
  the last 2 minutes; the Runs page then re-reads metrics every 5 s.
- *Playground:* models load inside the dashboard process on first use and stay loaded (at most
  two, least recently used dropped first; **Unload** frees them and the GPU). Generation streams
  over SSE from a POST (`fetch` + a stream reader, since EventSource can't POST); the Stop button
  aborts the request, the server's next write fails, and generation stops. Next-token panel (top
  10 with probabilities and entropy; click one to extend the prompt), side-by-side compare (two
  models or two snapshots; run one after the other on the shared GPU), chat mode with a
  per-model conversation for chat-format models (our base models refuse it with a clear message).
- *Data page / `sleuth data`:* sizes of `data/raw`, `data/finetune`, `data/shards`,
  `data/tokenizers`, with token counts and source mix from each shard folder's meta.json.
- *docs/DATA.md Stage 3:* the four fine-tuning/memory datasets the owner downloaded (MemGPT traces,
  Hermes FC v1, When2Call, LongMemEval) with licences: When2Call is CC BY 4.0 (credit it in
  anything published), LongMemEval MIT, the others Apache-2.0; Hermes/Glaive and When2Call are
  largely model-generated.
- *A plan correction:* the review's SFT step 1 ("add role tokens to the 32k tokenizer") is mostly
  done already. `<|system|>`, `<|user|>`, `<|assistant|>` and `<|end|>` have been reserved since M1,
  plus 9 spare slots for a tool role. No tokenizer retraining is needed.
- Tests: 181 → 227 CPU tests. New: the job runner with a fake `sleuth` (allow-list rejections,
  queueing, STOP-file and signal stops, history after a restart), the real trainer started,
  stopped with a checkpoint and resumed as jobs, the HTTP layer (token, Origin, Host, SSE logs,
  Playground generate/next/errors/GPU-busy), stale runs, the suite checks. The pages were also
  driven in headless Chromium (Playwright, outside the test suite): every page works at desktop
  and phone width, in light and dark, with no console errors.
- **Not yet tested on the owner's machines:** Ctrl-Break stopping on Windows (`CREATE_NEW_PROCESS_GROUP`),
  the Playground on the GPU (peak memory, GPU lane), loading Gemma 4 in the Playground, and the
  page in Edge/Chrome on Windows. Round 5 should start a short training job from the Jobs page,
  stop it, and resume it.

**2026-10-25: Round 5: mix-100m knows its first facts; Gemma at 10.3 tok/s; the gap to llama.cpp isn't bf16.**
Run by the owner's agent on the 5060 Ti box, code at `728495e` (before the knowledge-suite and
dashboard changes above, so the suite was still the 40-item one). All 181 CPU + 6 GPU tests passed.
- *Gemma 4 12B decoding* with the few-row kernel and fused norms: **7.7 → 10.3 tok/s**, prompt
  127 → 97 ms, peak 7.68 GiB. llama.cpp: 46, so still 4.5× slower. Next is CUDA graphs (§4.5.1).
- *First-token difference, fp32:* `next --dtype fp32` still puts "0" (30.1%) ahead of "1" (21.3%),
  by 0.344 logits (bf16: 0.375). So it is **not** bf16 rounding. New hypothesis: llama.cpp's own
  rounding. Its CUDA code doesn't multiply Q4_0 weights by the activations as they are: it first
  rounds the activations to 8 bits (q8_1, 32 values per fp16 scale) and takes integer dot products;
  its KV cache is fp16 too. Our kernel uses the activations unrounded, so llama.cpp is the one
  further from the exact result of the quantized weights. Experiment added: `next --q8-activations`
  (and `perplexity --q8-activations`) rounds every quantized layer's input the same way. If "1"
  then wins, that settles it; if not, a layer-by-layer comparison is next.
- *Corrected validation (re-sharded, 640 windows over the whole mix):* mix-20m **3.322** nats
  (1.139 bits/byte), mix-60m **2.971** (1.019). The 40-window numbers last round (3.34 / 2.99) were
  close; the old FineWeb-only number (3.90) was not. Python is the easiest source (1.85 for 60m),
  FineWeb-Edu the hardest (3.53).
- *mix-100m* (101M params, 76M non-embedding, 1.57B tokens): **10.4 h** at 65–72% MFU, 6.1 GB
  (config estimate ~12 h). Val **2.772 nats, 0.951 bits/byte**: cosmopedia 2.47, finemath 2.46,
  fineweb-edu 3.34, python-edu 1.68, sports 3.17, wikipedia 3.01. Every source improved by about
  the same amount from 60m (0.17–0.24 nats), so no single kind of text is holding it back.
- *The ladder:* 3.322 → 2.971 → 2.772. Per decade of training compute (6 · non-embedding params ·
  tokens: 2.8e16, 2.6e17, 7.1e17) that's −0.36, then −0.45 nats: no sign of flattening yet. If the
  trend holds, mix-150m on ~3B tokens (~2.2e18) should land near 2.55–2.6 (a prediction to check,
  not a promise: with three points and size and tokens growing together it's a rough line).
- *Knowledge, few-shot is what shows it.* Answer loss with `score --few-shot`: 2.945 → 2.502 →
  **1.943**; on facts alone 4.74 → 3.73 → **2.23**. Without examples, answer loss barely moved from
  60m to 100m (2.79 → 2.72), because then it's mostly measuring that a base model doesn't know to
  answer briefly. So for base models, few-shot answer loss is the number to track. That also
  answers last round's question: the facts answer loss does keep falling with size.
- *First real facts:* greedy few-shot answers right on 10/40 for mix-100m (60m: 4, 20m: 3), the
  first non-Python ones being Paris, Tokyo, Egypt and Wimbledon. Two of the ten were flukes, found
  by the owner's agent: "water freezes at" → "0.5 degrees Fahrenheit" counted as " 0", and "3
  strikes" came from a default guess: the model said "3" to soccer players, golf holes, a
  hexagon's sides and more. So ~8 genuine. Maths: 0/10, mostly echoing a number from the question.
- *Samples (greedy, no penalties):* "Photosynthesis is the process of converting sunlight into
  energy" (20m/60m had "carbon dioxide into carbon dioxide"); Fibonacci has the two-term recursion
  but a wrong base case. Every size still loops; `--no-repeat-ngram` (added since) is for that.
- *Induction heads:* random-token gain 0.16 at step 256, 2.33 at 500, 8.82 at 1000, 10.08 at the
  end (first copy 10.82 → repeat 0.75). This time **two layers** hold them: L5H7 (0.85), L5H6
  (0.77), L8H5 (0.75). (20m: layer 5 of 6; 60m: layer 4 of 10.) They formed between steps 256 and
  1000 at every size; 100m's longer warmup puts it a little later. *Logit lens:* final 2.619 /
  49.8% top-1; the biggest single step is L9 → L10 (−0.72 nats, +14 points).
- *Not Sleuth:* the agent's download filter also fetched LongMemEval `m` (2.6 GB), which isn't
  needed (docs/DATA.md: can be deleted).

**2026-10-25: After round 5: exact match needs the answer to end, a llama.cpp rounding experiment, document masking.**
- *Exact match (bug from round 5).* An answer now counts only if the model's own next token
  after it doesn't carry on the word or number: whitespace, an end-of-text/turn token, or
  punctuation followed by whitespace ends it; a letter or digit doesn't ("Paris" + "ian",
  "0" + ".5"). The token after the answer comes free from the scoring pass; only after punctuation
  does it look up to 3 tokens further (greedy). `--show-items` prints that continuation next to
  each match. Ablation still compares losses only, so it doesn't use this.
- *Default guesses.* `score` adds a note when one first answer token is the model's top guess in
  at least 20% (and 3+) of a category's tests, and how many of those count as exact matches. It
  skips a leading space token (some tokenizers split " 11" into " " + "11"). The tiny CPU demo
  model already shows it: "3" in 8 of 12 math tests.
- *`--q8-activations`* for `next` and `perplexity` (see round 5 above). The rounding matches our
  Q8_0 quantizer to within one step (a test checks it), and only applies to layers kept quantized.
- *Document masking (M11a, step 1).* `train.doc_mask = true` gives the model each window's
  document numbers (`document_ids`: +1 at every `<|bos|>`; every shard document starts with one,
  so this finds the same boundaries as the `.idx.npy` files without reading them). Attention then
  ANDs the causal mask with "same document": one causal triangle per document. Decisions:
  - Positions keep counting across the boundary. RoPE scores depend only on query–key distance,
    so a masked document gives the same logits wherever it starts (a test checks exactly that:
    the second document in a masked window equals the same text on its own).
  - The `<|eos|>` belongs to the document it ends (it's predicted from that text); the next
    `<|bos|>` only sees itself.
  - With a mask, SDPA can't use FlashAttention and uses its memory-efficient kernel. K/V heads
    are repeated by hand in that case, because that kernel may not accept `enable_gqa` and SDPA
    would then fall back to the math kernel, which stores the whole attention matrix. The speed
    cost on the 5060 Ti is unmeasured; FlexAttention is the fallback if it's large.
  - Validation follows the run's setting; `sleuth eval RUN --doc-mask yes/no` overrides it (on a
    run trained without masking, that measures how much the model leans on the previous
    document). CPU demo model: 0.538 → 0.544 nats with the mask.
- Tests: 227 → 243 CPU tests; GPU tests 6 → 8 (masking under bf16, with and without torch.compile).

**2026-10-25: Round 6: masking helps a little, the space-token trap in few-shot scoring, mix-150m planned.**
Run by the owner's agent on the 5060 Ti box at `9f8ff84` (PR #7). The owner checked the
dashboard by hand: Jobs (a training job streamed, stopped with a checkpoint and resumed),
Playground (mix-100m; Gemma 4 in chat mode) and the logit-lens note all work.
- *Three Windows-only test failures, fixed by the owner's agent (commit `57b5ba3`):* a test wrote a
  file with `write_text` (`\r\n` makes it 13 bytes, not 12); `read_log` with a negative offset always
  dropped the first line, even when the window started exactly at one (`finished\r\n` is 10 bytes);
  and **a Ctrl-Break sent right after a process starts is lost on Windows**: the dashboard now
  re-sends it every second, up to 5 times. Plus a Playground fix: the model box's `<datalist>` only
  suggested runs matching the pre-filled text, so a Run `<select>` now lists every run. Follow-up
  here: the trainer ignores every stop signal after the first, so a re-sent one can't interrupt the
  checkpoint the first one is saving.
- *Gemma's first token: llama.cpp's activation rounding is not the reason.* With
  `--q8-activations` the margin of "0" over "1" only shrinks from 0.344 to 0.333 logits; perplexity
  moves 22.461 → 22.487 (llama.cpp 22.59). I'm parking this: it's one position of raw text, where
  Gemma IT is badly out of its element (~19 nats/token at positions 1–7, round 3) and small
  differences get amplified, while perplexity agrees within 0.6% and the chat-format answers agree.
  If it matters later: llama-server's `n_probs` for the same prompt, then a layer-by-layer dump.
- *The ladder on the 220-item suite* (answer loss / exact; few-shot answer loss / exact):
  mix-20m 3.280 / 3.2%, 3.241 / 3.6%; mix-60m 2.856 / 4.5%, 2.801 / 3.6%; mix-100m 2.803 / 4.5%,
  **2.112 / 14.1%**. mix-100m few-shot gets 10 capitals, 1914, French, Wimbledon, FIFA, Cricket,
  Tennis and Brazil right. The "answer must end" rule removed round 5's "0.5 degrees" match and
  kept "32" + " degrees". Several "3"s are default guesses, as the notes said.
- *A scoring trap, found from those notes:* in few-shot, mix-20m's "default guess" for facts was
  a bare " " in 54 of 70 tests. Our tokenizer writes numbers without their space (" 7" = " " +
  "7"), and two of the four `_fewshot.txt` answers are numbers, so the prompt teaches "after `A:`
  comes a lone space". A model that then writes " " + "Paris" has the right text, but we scored
  only the one-token spelling " Paris" (and its top-1 " " failed exact match). That hit the small
  models hardest, so "few-shot only helps at 100M" may be partly an artifact. Fixed (below);
  the ladder needs one more re-score.
- *Document masking, speed (mix-60m, 300 steps):* 63.0k → 53.9k tok/s (−14.5%, MFU 65 → 56%), but
  less memory (5.7 → 4.8 GB). The SDPA kernel that accepts a mask is slower than FlashAttention.
- *Document masking, A/B (mix-20m, same 0.5B tokens; 62 min with compile vs 95 min before):*

  | val loss (640 windows) | eval unmasked | eval masked |
  |---|---|---|
  | mix-20m, trained unmasked | **3.3221** | 3.3302 |
  | mix-20m-docmask, trained masked | 3.3500 | **3.3095** |

  Each model is best measured the way it was trained, and masked/masked beats unmasked/unmasked by
  0.013 nats (0.0044 bits/byte) on every source. The masked model loses more when the mask is taken
  away (+0.041) than the unmasked one when it's added (+0.008): the unmasked model barely used the
  previous document anyway, while the masked one never learned to ignore it. Caveat (the agent's):
  the runs also differ in micro-batch (16×4 vs 8×8), so they saw different batches, and run-to-run
  noise at this size is probably of the same order; the direction on all six sources is the
  evidence. Knowledge (few-shot score) is the same within error bars. Decision: mix-150m trains
  masked (cheap, slightly better, and the chat fine-tune needs it). Unexplained: the masked run's
  peak memory grew from 5.1 to 7.8 GB during the run.
- *Data:* without caps the downloaded files hold ~20 GB of text (≈ 4.7B tokens), but Cosmopedia
  (1.4 GB) limits a mix with the same shares to ~2.2B. With one more Cosmopedia file, ~3.6B
  (docs/DATA.md, "A bigger mix for mix-150m").

**2026-10-25: After round 6: scoring the answer text, FlexAttention for masks, mix-150m config.**
- *Scoring the text, not one token sequence* (`score`; ablation keeps the token-level version):
  - **Answer loss** counts both spellings, P(" Paris") + P(" " + "Paris"), normalized by the
    canonical token count as before. Numbers (" 7") have only one spelling, so they don't change.
    `score` reports per category how much probability came through the " " + word spelling.
  - **Exact match** is now the model's *greedy answer*, decoded and compared as text, which
    must also end there (whitespace, an end token, or punctuation then whitespace). It replaces
    last round's "top-1 at every answer token, then look ahead". It stops as soon as the outcome is
    known (≤ answer length + 4 tokens, with a KV cache). Cost: on the tiny test Gemma,
    `score --few-shot` went 2.4 → 5.9 s; on the 12B, expect a few minutes per suite.
  - `--show-items` prints the greedy answer, and the default-guess note now counts the first *word*
    of the greedy answers ("3", "The"), not the first token (" " was a spelling, not a guess).
- *FlexAttention for document masks:* `train.doc_mask_kernel = "flex"` builds a BlockMask per
  micro-batch (outside the compiled model) and runs `flex_attention`, which torch.compile turns
  into a Triton kernel that skips fully masked 128×128 tiles. Checked on CPU against the SDPA path
  (same logits to 1e-5); it has no backward pass on CPU, so the trainer refuses it there, before
  making a run folder. GPU speed and Windows support (triton-windows) are untested: that's round 7.
- *`configs/mix-150m.toml`:* d=896, 14 layers, 14 heads (head size 64), 7 KV heads: 155M params
  (125M non-embedding), 23,000 steps × 131k = 3.0B tokens, 1082 MFLOPs/token, ~30 h at 68% MFU
  (~35 h with the SDPA mask). Same LR (6e-4) and cosine schedule as mix-100m, document-masked,
  shards `data/shards/mix-3b`.
- Tests: 243 → 248 CPU; GPU 8 → 10 cases (FlexAttention; the case without compile skips).

**2026-10-25: MoE layers, and MTP heads for our own models (GPU box, round 7 follow-up).**
- *MoE* (`forge/model/moe.py`, from llama.cpp's build_moe_ffn): softmax router in fp32, top-k,
  weights renormalized; experts stay as the file's quantized blocks and only the chosen ones run
  (a Python loop over experts: exact, not fast). `qwen35moe` (Qwen3.6-35B-A3B: 256 experts, 8 used,
  plus a sigmoid-gated shared expert; its MTP block is MoE too) and Gemma 4 26B-A4B (128 experts,
  8 used, router on rms(x)/√d × scale, the dense MLP kept as a shared expert). Checked on tiny
  GGUFs against loop-by-loop references; the real files wait for offloading (21 GB / 15.8 GB).
- *MTP heads* (`model.mtp_heads`, `forge/model/mtp_heads.py`): head k guesses the token k+1 ahead
  from the previous head's state and the embedding of the token before its target (DeepSeek-V3's
  chain), sharing the output layer. Decision: **no attention in the heads**, so drafting needs no KV
  cache and no catch-up after each check (Hydra-style); attention is a later experiment. Loss =
  main + `train.mtp_weight` (0.3) × mean head loss; the logged `loss` stays the next-token loss, so
  runs compare with older ones, and `mtp_loss` is logged next to it. `--draft mtp` and the
  Playground use a run's own heads. Cost on mix-150m's shape: +21% FLOPs per head (mostly the
  32k-vocab output layer again), +8M parameters. Plan: one head on the run after mix-150m.
- CUDA graphs are refused cleanly for qwen35 and MoE models (`graph_unsupported`); the Playground
  used to fail loading qwen35 on a GPU because its warm-up always recorded a graph.
- Tests: 266 → 276 CPU.

**2026-10-25: Fewer kernels per token (fused norms and RoPE).**
- Baseline from the owner's llama.cpp server on an identical RTX 5060 Ti, same Gemma 4 12B file
  (the "QAT Q4_K_XL" file is 100% Q4_0) with MTP: 108-112 tok/s, ~25 ms per pass of ~2.8 tokens.
  Ours: 70 tok/s, ~38 ms per pass. Same hardware and file, so the gap is our engine.
- Counted on the tiny Gemma's structure (one static decode step): **100 kernels per layer**, 27 of
  them dtype casts. Each RMSNorm was ~9 kernels (the fp32 casts of x and the gain, square, mean,
  eps, rsqrt, two multiplies, cast back) and Gemma has 6 per layer, plus RoPE (~7 per tensor) and
  the value norm. The static attention mask was rebuilt in every layer.
- New `forge/model/fused_ops.py`: one Triton kernel per RMSNorm, and one per q/k/v tensor for the
  head norm + RoPE. The static cache builds each mask and the positions once per pass. Now **29
  kernels per layer**, under 1 cast. For the 12B that is ~4,800 -> ~1,400 kernels per token.
  Speed is unmeasured until the GPU is free (training).
- Inference only: used when autograd is off, on CUDA (or the Triton interpreter in tests), outside
  torch.compile, after a one-time self-check; training and CPU runs keep the eager code. One
  numeric difference: the fused kernel doesn't round the normalized q/k to bf16 before RoPE (one
  rounding instead of two, as in llama.cpp). Rows are independent, so speculative checks stay
  exact.
- Tests: 276 -> 277 CPU (interpreter: kernels vs eager, whole models vs eager and vs the llama.cpp
  reference, gradients still flow in training) + 1 GPU test (bf16, graphs = eager).

**2026-10-26: mix-150m finished; fused kernels measured.**
- mix-150m (154.6M params, 23,000 steps, ~3.0B tokens of mix-3b): **val bpb 0.8813**, against
  0.9514 for mix-100m (-7.4%). Knowledge suite: answer loss 2.803 -> 2.502 plain, 2.112 -> 1.775
  few-shot; few-shot exact match 14.1% -> 18.2% (facts 27%: most capitals right). Greedy
  "12 times 12" now gives 144, the first correct arithmetic in the ladder; fibonacci has both
  base cases right. It still loops after a sentence.
- Gemma 4 12B decode, warm runs, fused kernels off -> on: eager 8.7 -> 14.4 tok/s, CUDA graph
  36.5 -> 42.5 (27.4 -> 23.5 ms per token), MTP + graph 72.1 -> 71.5 (no change: the draft
  chain and the 4-row verify now dominate, ~36.6 ms per pass against llama.cpp's ~25). All modes
  give exactly the greedy text at 3 guesses per pass.
- Surprise: the bf16 GPU tests failed, fused vs eager differing by ~15% of the logits. The kernels
  were right (fp32: 3e-5). The tiny random test model magnifies the one rounding the two paths
  do differently ~70x over 6 layers. At Gemma's real head size the fused kernel is closer to exact
  fp32 math (RMS error 0.0017 vs 0.0024). The tests now check bf16 precision on the kernel itself,
  and the matmul-kernel test keeps the norm kernels fixed so it compares only the matmul.
- Lesson for benchmarks: a single cold run (graph capture, Triton compiles) under-reports speed
  by ~25%; always warm up first.
