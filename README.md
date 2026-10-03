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
* **Time-to-performance benchmark** (`benchmarks/ttp.py`) with committed results, see below.
* Breaking: Python >= 3.10, `sakura-ml>=1.1`, the unused `zakuro-ai` dependency is dropped and the
  trainer constructor changed (see [CHANGELOG](CHANGELOG.md)).

# Benchmark: time to performance

Epoch time is the wrong metric once evaluation and checkpoints can be overlapped or throttled.
What matters is **how soon a model of the required quality exists and is known to exist**.
`benchmarks/ttp.py` trains the same DeepSpeech2 (GRU 3x512) on a learnable synthetic task
(each character is a spectral prototype plus noise) with the same seed, and stops the first time a
*resolved* evaluation reaches the target CER, so the measured time includes evaluation lag.

| arm | what runs |
|---|---|
| `legacy` | the 0.4 loop: synchronous evaluation, the best model is written only when it improves |
| `vanilla+ckpt` | the same loop plus a resume checkpoint after every epoch (what a crash-safe run needs) |
| `sakura-thread` | Sakura runtime; evaluation and checkpoint writes on background threads |
| `sakura-process` | Sakura runtime; background work in a worker process (shared-memory tensor hand-off) |
| `sakura-process-30s` | as above, resume checkpoints at most every 30 s (`checkpoint_every_s`) |

RTX 2080 Ti on a shared host, torch 2.14.1+cu130, fp16, target CER 15, seed 0
(`benchmarks/results/ttp-long.json`; a 44-epoch run):

| arm | time to CER 15 | vs `vanilla+ckpt` | vs `legacy` | mean epoch |
|---|---:|---:|---:|---:|
| `legacy` | 104.0 s | 1.80x | 1.00x | 2.37 s |
| `vanilla+ckpt` | 186.9 s | 1.00x | 0.56x | 4.33 s |
| `sakura-process` | 102.0 s | 1.83x | 1.02x | 2.29 s |
| `sakura-process-30s` | 98.1 s | **1.91x** | **1.06x** | 2.22 s |

Short runs (about 10-14 epochs, 3 seeds, `ttp-short-seed*.json`), median time to target:
`legacy` 25 s, `vanilla+ckpt` 49 s, `sakura-thread` 39 s, `sakura-process` 42 s, `sakura-process-30s` 29 s.

What this does and does not show:

* **Crash safety is expensive in a synchronous loop** (+80% wall-clock for a per-epoch resume
  checkpoint) and **Sakura makes it nearly free**: about 1.8-1.9x faster than the safe synchronous
  loop to the same quality.
* **It is not faster than the old unsafe loop.** Against `legacy` it is on par (1.02-1.06x in the
  long run, 0.5-1.07x across short runs). `legacy` simply writes nothing but the best model.
* **Threads lose to a worker process.** Checkpoint serialisation in a thread competes for the GIL
  with the launch-bound training loop (training epochs became 2.4x slower in a profile), which
  cancels the overlap.
* **Fixed costs show on short runs:** about half a second of worker start-up, and evaluation lags
  training by one epoch. The shared host also makes single runs noisy (a seed's arms differ by up to
  2x); treat the short-run table as indicative and the long run as the reference.
* The task is synthetic, so this measures systems behaviour, not accuracy.

Reproduce: `python benchmarks/ttp.py --device cuda --target 15 --max-epochs 120 --noise 4.0 --lr 3e-5 --train-size 512 --eval-size 1024`.
`python benchmarks/run.py` is the earlier epoch-throughput benchmark (random targets).

# Installation

```bash
pip install asr-deepspeech            # pulls sakura-ml>=1.1
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
