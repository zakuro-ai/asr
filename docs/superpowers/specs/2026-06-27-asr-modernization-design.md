# asr_deepspeech Modernization — Design

- **Date:** 2026-06-27
- **Status:** Approved (pending spec review)
- **Scope:** Full modernization — device portability, deprecation fixes, dead-code/visdom removal, `eval()` removal, type hints, expanded tests, packaging tidy.
- **Delivery:** One cohesive PR on `feat/modernize-device-portability`, logically committed, CI green at each step.

## Context & problem (verified)

Verified in an isolated git worktree + fresh `uv` venv on a CPU-only host (no GPU
in the container or on i9):

- `uv sync` clean; `pytest` → **62 passed**; torch 2.12.1, Python 3.12.
- **Training cannot run out of the box.** `config.yml` defaults `device: cuda`,
  so `model.to("cuda")` raises `RuntimeError: Found no NVIDIA driver`. The
  codebase is effectively CUDA-only.
- `torch.cuda.amp.*` is **deprecated** and auto-disables when CUDA is absent.
- When forced to CPU, the model math is sound: a real 832K-param DeepSpeech model
  ran 80 CTC steps with loss **54.33 → 2.58** and the greedy-decode path runs.

So the modeling is fine; the **defaults, device handling, and several torch
APIs** are stale. This blocks the requested "short training cycle" on any
GPU-less environment.

## Goals

1. Make training and inference **device-agnostic** (auto-detect CPU/GPU; honor
   explicit overrides). Training must run on CPU.
2. Remove deprecated torch APIs (AMP, tensor constructors, `dist.reduce_op`).
3. Remove `eval()` from the model/config path.
4. Delete dead code (`models/deepspeech_model.py`) and unmaintained visdom.
5. Add type hints to core methods.
6. Expand tests, keep the ≥50% coverage gate green.

## Non-goals

- **No modeling/numerics changes.** GPU runs stay bit-for-bit unchanged.
- No retraining of released checkpoints; no new model architectures.
- No change to the public training entry point (`python -m asr_deepspeech.trainers`).
- Not an exhaustive type-hint sweep — core methods only.

## Design — 7 units

### Unit 1 — Device portability (anchor)
- New `asr_deepspeech/device.py`: `resolve_device(spec) -> torch.device`.
  - `"auto"` → `cuda` if `torch.cuda.is_available()` else `cpu` (warn once on fallback).
  - `"cuda"`/`"cpu"` honored explicitly; unknown → error.
- `config.yml`: `device: auto`, `device_test: auto` (string `"cuda"`/`"cpu"` still valid).
- `vars.py:14`: guard `torch.cuda.manual_seed_all(...)` behind `torch.cuda.is_available()`.
- `modules/deepspeech.py` `__call__`: stop defaulting `device="cuda"`; resolve/accept the passed device.

### Unit 2 — Device-agnostic AMP
- `trainers/deepspeech_trainer.py:69,75`: `torch.cuda.amp.GradScaler()` /
  `torch.cuda.amp.autocast()` → `torch.amp.GradScaler(device_type)` /
  `torch.amp.autocast(device_type)`.
- AMP enabled **only** when `device.type == "cuda"`; on CPU, `mixed_precision`
  becomes a safe no-op (preserves GPU behavior exactly).

### Unit 3 — Replace deprecated tensor/dist APIs
- `functional.py:19-20,31`: `torch.FloatTensor()/IntTensor()` → `torch.zeros(..., dtype=...)` / `torch.tensor(..., dtype=...)`.
- `functional.py:38`: `dist.reduce_op.MAX/SUM` → `dist.ReduceOp.MAX/SUM`.
- `data/parsers/spectrogram_parser.py:55`: `torch.FloatTensor(spect)` → `torch.from_numpy(...)`/`torch.as_tensor(..., dtype=torch.float32)`.
- `modules/blocks.py:50-52`: `torch.BoolTensor(...).fill_(0)` + `.cuda()` → `torch.zeros(x.shape, dtype=torch.bool, device=x.device)`.

### Unit 4 — Remove `eval()`
- `modules/deepspeech.py:43`: `eval(rnn_type)` → `SUPPORTED_RNNS` mapping
  accepting both `"nn.LSTM"` and `"lstm"` (backward compatible).
- `trainers/__main__.py:34`: `eval(cfg.optim.betas)` → `ast.literal_eval`,
  accepting both `"(0.9, 0.999)"` and a native YAML list.

### Unit 5 — Remove dead code + visdom
- Delete `models/deepspeech_model.py` (imports a non-exported `VisdomLogger`; the
  whole `models` package `ImportError`s if touched — tests already skip it).
- Remove the dead `models/__init__.py` import (delete the package if it ends up empty).
- Delete `loggers/visdom_logger.py`; remove `--visdom` arg + visdom mention in `parsers/default_parser.py`.
- `decoders/greedy_decoder.py:2`: `from six.moves import xrange` → `range` (drops `six`).
- Update `tests/test_imports.py` skip list accordingly.

### Unit 6 — Type hints (core methods only)
- `DeepSpeech.forward / get_seq_lens / __call__`, `DeepSpeechTrainer.fit / train / test`,
  `device.resolve_device`, `GreedyDecoder`/`Decoder` public methods.

### Unit 7 — Test coverage + packaging tidy
- New tests:
  - `device.resolve_device`: `auto`→cpu fallback, explicit cpu/cuda, bad spec errors.
  - `rnn_type` lookup: `"nn.LSTM"` and `"lstm"` resolve to `nn.LSTM`.
  - **CTC train step**: synthetic batch, a few steps, assert loss **decreases**
    (ports the verification harness into the suite).
  - `blocks.py` mask: correct shape/dtype/device on CPU.
- Keep ruff config as-is; keep coverage gate ≥50% (nudge up if headroom).
- Confirm `pyproject.toml` has no stale deps after `six` removal.

## Key decisions
- **Backward compatible:** existing `cuda` / `nn.LSTM` / `"(0.9,0.999)"` config
  values keep working; `auto` is only the new default.
- **AMP gated to CUDA:** no CPU bf16 autocast, to keep GPU runs unchanged.
- **visdom removed entirely** (already non-functional).
- **GPU numerics unchanged** — every change is portability/deprecation/cleanup.

## Testing & verification
- CI gates: `ruff check` + `ruff format --check` + `pytest` + `--cov-fail-under` (≥50%).
- Final evidence: re-run the short **CPU training cycle** (loss-decreases + decode)
  in the isolated worktree.
- GPU smoke-test: documented but not runnable from here (no GPU available).

## Risks & mitigations
- *AMP API change*: gate on `device.type`; covered by existing trainer tests + new step test.
- *Removing `models`/visdom*: nothing imports them at runtime (verified via grep);
  `test_imports` updated.
- *`betas` parsing*: `ast.literal_eval` handles the legacy string and YAML list;
  add a unit test.

## Delivery
- Branch `feat/modernize-device-portability` → PR to `master`.
- Commits grouped by unit; CI green per commit.
- Verified by the CPU training cycle before merge.
