<h1 align="center">
  <br>
  <img src="https://drive.google.com/uc?id=17SeD6ijR7DV_EnZGJqavHxVbNHs8n4EQ">
  <br>
    ASRDeepspeech x Sakura-ML 
    (English/Japanese)
  <br>
</h1>

<p align="center">
  <a href="#whats-new-in-05">What's new</a> •
  <a href="#benchmark">Benchmark</a> •
  <a href="#modules">Modules</a> •
  <a href="#code-structure">Code structure</a> •
  <a href="#installing-the-application">Installing the application</a> •
  <a href="#makefile-commands">Makefile commands</a> •
  <a href="#environments">Environments</a> •
  <a href="#dataset">Dataset</a>•
  <a href="#running-the-application">Running the application</a>•
  <a href="#notes">Notes</a>•
</p>


DeepSpeech2 speech recognition (English / Japanese) in PyTorch, trained with the
[Sakura](https://github.com/zakuro-ai/sakura) runtime. A clean, modular take on SeanNaren's
implementation (trainers, models, loggers, decoders), driven by a single YAML configuration.
A pretrained Japanese model reaches `CER = 34` on the JSUT test set.

# What's new in 0.5

* **Sakura 1.0 trainer.** `DeepSpeechTrainer` is now a plain class that drives a `SakuraRuntime`
  through a `DDPAdapter`: `MixedPrecision` (autocast + GradScaler), `AsyncEval` (evaluation
  overlapped with the next epoch, with an adaptive gate that measures both modes and keeps the
  faster one) and `AsyncCheckpoint` (atomic rolling checkpoints written off the training thread).
* **Safer checkpoints.** New checkpoints are plain tensors/dicts (`torch.load(weights_only=True)`),
  written atomically; checkpoints from <= 0.4 still load. A resumed run continues from the newest
  rolling checkpoint.
* **Fewer silent failures.** Invalid-loss batches are counted and logged, a run where *every*
  batch is invalid aborts, failed asynchronous evaluations are reported, and the import-time global
  RNG seeding is gone (use `seed:` in the config).
* **Reproducible benchmark** (`benchmarks/run.py`) with committed results, see below.
* Breaking: Python >= 3.10, `sakura-ml>=1.0`, the unused `zakuro-ai` dependency is dropped and the
  trainer constructor changed (see [CHANGELOG](CHANGELOG.md)).

# Benchmark

`python benchmarks/run.py` trains the same DeepSpeech2 (GRU 3x512) on deterministic synthetic
spectrograms with the same seed, varying only the runtime:

| arm | what runs |
|---|---|
| `vanilla` | synchronous evaluation and checkpoint writes, hand-rolled GradScaler (0.4 behaviour) |
| `sakura-sync` | Sakura runtime, async checkpoint, synchronous evaluation |
| `sakura-async` | Sakura runtime, adaptive async evaluation + async checkpoint |

RTX 2080 Ti, torch 2.14.1+cu130, fp16 autocast, 10 epochs (raw JSON in `benchmarks/results/`):

| workload | vanilla | sakura-sync | sakura-async | best CER |
|---|---:|---:|---:|---|
| 512 train / 512 eval utterances | 37.0 s | 36.0 s (1.03x) | **33.9 s (1.09x)** | 92.87 for all arms |
| 512 train / 2048 eval utterances | 57.8 s | 54.8 s (1.05x) | 55.6 s (1.04x) | 92.50 / 92.55 / 92.55 |

How to read this honestly:

* The gain is **modest (4-9%)** and, with one run per arm, partly within run-to-run noise
  (about +/-3% between epochs). Error rates match: Sakura changes *when* work happens, not what.
* Evaluation is overlapped on a thread, so the part of `DeepSpeech.__call__` that is plain
  Python (greedy decoding, Levenshtein) still competes for the GIL with the training loop.
  Moving evaluation to a Sakura worker process is the obvious next step and is not done yet.
* The final epoch can never overlap its own evaluation, so short runs under-sell the effect.
* The data is synthetic, so CER stays near chance; this benchmark measures throughput, not
  accuracy. A CPU run on a shared laptop was too noisy to report and is deliberately not included.

Reproduce: `python benchmarks/run.py --epochs 10 --device cuda --train-size 512 --eval-size 512 --batch-size 32 --hidden 512 --layers 3`.

# Installation

```bash
pip install asr-deepspeech            # pulls sakura-ml>=1.0
# or from source
git clone https://github.com/zakuro-ai/asr && cd asr
uv sync --extra test
```

Python >= 3.10 and [PyTorch](https://pytorch.org/get-started/locally/) are required. Docker images
are available through `make docker-sandbox`.

# Quickstart

```bash
python -m asr_deepspeech.etl                              # download + prepare JSUT
python -m asr_deepspeech.trainers                         # train (Sakura runtime)
python -m asr_deepspeech.trainers --runtime vanilla       # synchronous reference loop
python -m asr_deepspeech.trainers --no-async-eval         # Sakura, synchronous evaluation
python -m asr_deepspeech                                  # evaluate the pretrained model
```

From Python:

```python
import torch
from torch.nn import CTCLoss
from asr_deepspeech import cfg
from asr_deepspeech.modules import DeepSpeech
from asr_deepspeech.trainers import DeepSpeechTrainer

model = DeepSpeech(**vars(cfg.model))
train_loader, _ = model.get_loader(
    manifest=cfg.loaders.train_manifest, batch_size=48, num_workers=8
)
test_loader, _ = model.get_loader(manifest=cfg.loaders.val_manifest, batch_size=48, num_workers=8)
optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-4)

trainer = DeepSpeechTrainer(
    model,
    CTCLoss(reduction="sum"),
    optimizer,
    epochs=100,
    model_path="gold/model.pth",
    runtime="sakura",
    async_eval=True,
)
metrics = trainer.run(train_loader, test_loader)  # -> Metrics(best_cer, best_epoch, history, ...)
```

`model_path` receives the best-CER model; per-epoch resume checkpoints go to
`<model_path dir>/checkpoints/` (two newest kept). Re-running the same command resumes.

# Configuration

Everything is driven by `asr_deepspeech/config.yml`, loaded into the global `cfg`. Point
`ZAK_ASR_CONFIG` at your own YAML to override it. Trainer keys:

| key | default | meaning |
|---|---|---|
| `runtime` | `sakura` | `sakura` or `vanilla` |
| `async_eval` | `true` | overlap evaluation with training (adaptive; `sakura` only) |
| `rolling_checkpoints` | `true` | keep resumable per-epoch checkpoints |
| `mixed_precision` | `true` | autocast on CUDA |
| `device`, `device_test` | `auto` | training / evaluation device |
| `seed` | `123456` | RNG seed (`null` to disable) |
| `overwrite_lr` | `null` | override the learning rate after a restore |

# Modules

| Component | Description |
| ---- | --- |
| `asr_deepspeech.trainers` | `DeepSpeechTrainer` (Sakura / vanilla runtimes) and the training CLI |
| `asr_deepspeech.checkpoint` | atomic, `weights_only`-safe checkpoints; legacy loader |
| `asr_deepspeech.metrics` | `Metrics` / `EvalResult` run bookkeeping |
| `asr_deepspeech.modules` | DeepSpeech2 network |
| `asr_deepspeech.data` | datasets, loaders, parsers, samplers, synthetic data |
| `asr_deepspeech.decoders` | greedy / beam decoders |
| `asr_deepspeech.etl` | dataset download and manifests |
| `benchmarks/` | runtime benchmark and committed results |

# Development

```bash
make test                       # pytest
uv tool run ruff check . && uv tool run ruff format --check .
```
