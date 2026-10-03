"""Benchmark the legacy-style synchronous trainer against the Sakura runtime.

Same model, data, seed and epochs for every arm; only the runtime changes:

  vanilla        synchronous eval + synchronous checkpoint (reference, = pre-0.5 behaviour)
  sakura-sync    Sakura runtime, async eval disabled (async checkpoint only)
  sakura-async   Sakura runtime, adaptive async eval + async checkpoint

    python benchmarks/run.py --epochs 8 --out benchmarks/results/cpu.json

Reports seconds per epoch, total wall-clock, speed-up vs. vanilla and best CER
(the CER of every arm must match: Sakura changes *when* work happens, not what).
"""

from __future__ import annotations

import argparse
import json
import platform
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

import pandas as pd
import torch
from torch.nn import CTCLoss

from asr_deepspeech.data.synthetic import synthetic_loader
from asr_deepspeech.modules.deepspeech import DeepSpeech
from asr_deepspeech.trainers import DeepSpeechTrainer

ARMS = {
    "vanilla": dict(runtime="vanilla"),
    "sakura-sync": dict(runtime="sakura", async_eval=False),
    "sakura-async": dict(runtime="sakura", async_eval=True),
}

AUDIO_CONF = SimpleNamespace(
    sample_rate=16_000,
    window_size=0.02,
    window_stride=0.01,
    window="hamming",
    speed_volume_perturb=False,
    spec_augment=False,
    noise_dir=None,
    noise_prob=0.4,
    noise_levels=(0.0, 0.5),
)


def run_arm(name: str, args: argparse.Namespace, workdir: Path) -> Dict[str, Any]:
    torch.manual_seed(args.seed)
    labels = workdir / "labels.csv"
    pd.DataFrame({"label": list("abcdefghijklmnopqrstuvwxyz")}).to_csv(labels, index=False)
    model = DeepSpeech(
        audio_conf=AUDIO_CONF,
        decoder=None,
        label_path=str(labels),
        rnn_type="nn.GRU",
        rnn_hidden_size=args.hidden,
        rnn_hidden_layers=args.layers,
        bidirectional=True,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    train = synthetic_loader(args.train_size, args.batch_size, seed=1)
    test = synthetic_loader(args.eval_size, args.batch_size, seed=2)
    arm_dir = workdir / name
    trainer = DeepSpeechTrainer(
        model,
        CTCLoss(reduction="sum", zero_infinity=True),
        optimizer,
        epochs=args.epochs,
        model_path=str(arm_dir / "best.pth"),
        output_file=str(arm_dir / "output.txt"),
        device=args.device,
        device_test=args.device,
        mixed_precision=True,
        seed=args.seed,
        **ARMS[name],
    )
    Path(arm_dir).mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    metrics = trainer.run(train, test)
    total = time.perf_counter() - t0
    svc = trainer._async_eval_svc
    return {
        "arm": name,
        "total_s": round(total, 3),
        "epoch_s": [round(s, 3) for s in trainer.epoch_seconds],
        "best_cer": metrics.best_cer,
        "best_epoch": metrics.best_epoch,
        "eval_modes": [m for _, m in svc.decisions] if svc is not None else None,
        "skipped_batches": metrics.skipped_batches,
    }


def markdown(results: List[Dict[str, Any]], meta: Dict[str, Any]) -> str:
    base = next(r for r in results if r["arm"] == "vanilla")["total_s"]
    lines = [
        f"Device `{meta['device']}` · torch {meta['torch']} · {meta['epochs']} epochs · "
        f"{meta['train_size']} train / {meta['eval_size']} eval utts · "
        f"GRU {meta['layers']}x{meta['hidden']}",
        "",
        "| arm | total (s) | speed-up | best CER |",
        "|---|---:|---:|---:|",
    ]
    for r in results:
        lines.append(
            f"| {r['arm']} | {r['total_s']:.1f} | {base / r['total_s']:.2f}x "
            f"| {r['best_cer']:.2f} |"
        )
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--device", default="auto")
    p.add_argument("--train-size", type=int, default=256)
    p.add_argument("--eval-size", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    results = []
    with tempfile.TemporaryDirectory() as tmp:
        for name in args.arms:
            print(f"== {name}", flush=True)
            results.append(run_arm(name, args, Path(tmp)))
            print(json.dumps(results[-1]), flush=True)

    from asr_deepspeech.device import resolve_device

    meta = dict(
        device=str(resolve_device(args.device)),
        torch=torch.__version__,
        platform=platform.platform(),
        epochs=args.epochs,
        train_size=args.train_size,
        eval_size=args.eval_size,
        layers=args.layers,
        hidden=args.hidden,
    )
    print("\n" + markdown(results, meta))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"meta": meta, "results": results}, indent=2))


if __name__ == "__main__":
    main()
