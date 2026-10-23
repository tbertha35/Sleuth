# Hardware probe results

`scripts/probe_hardware.py` writes one JSON file per machine here. **These files
are git-ignored and stay on your machine**: they hold machine details (host name,
drive paths) that don't belong in a public repo. The probe already leaves out
the GPU's serial-like UUID and shows your home folder as `~`.

Design decisions cite the numbers through anonymized summaries in `LOG.md`
(e.g. "RTX 5060 Ti: bf16 matmul 42 TFLOP/s").

Re-run the probe after driver or PyTorch upgrades with `--tag <reason>`, so old
and new numbers sit side by side.
