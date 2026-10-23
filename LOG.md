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
