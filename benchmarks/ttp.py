"""Time-to-performance benchmark: wall-clock until CER <= target on a learnable task.

Epoch time is the wrong metric when evaluation and checkpoints can be overlapped or
throttled: what matters is how soon a model of the required quality exists *and is known
to exist*. Each arm trains the same learnable synthetic task (seed, model, data identical)
and stops the first time a resolved evaluation reaches ``--target`` CER. The reported time
therefore includes evaluation-detection lag (up to one epoch with async evaluation).

Arms (see ARMS): the 0.4-style synchronous loop, the same loop with per-epoch resume
checkpoints, and Sakura with thread / process background workers.

    python benchmarks/ttp.py --device cuda --target 10 --out benchmarks/results/ttp.json
"""

from __future__ import annotations

import argparse
import json
import platform
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import torch
from run import AUDIO_CONF
from torch.nn import CTCLoss

from asr_deepspeech.data.synthetic import synthetic_loader
from asr_deepspeech.device import resolve_device
from asr_deepspeech.modules.deepspeech import DeepSpeech
from asr_deepspeech.trainers import DeepSpeechTrainer

ARMS: Dict[str, Dict[str, Any]] = {
    # 0.4 behaviour: synchronous evaluation, model written only when it improves
    "legacy": dict(runtime="vanilla", rolling_checkpoints=False),
    # same loop + a resume checkpoint after every epoch (what a crash-safe run needs)
    "vanilla+ckpt": dict(runtime="vanilla", rolling_checkpoints=True),
    "sakura-thread": dict(runtime="sakura", dispatch="thread", async_eval=True),
    "sakura-process": dict(runtime="sakura", dispatch="process", async_eval=True),
    # crash safety bounded by time instead of epochs
    "sakura-process-30s": dict(
        runtime="sakura", dispatch="process", async_eval=True, checkpoint_every_s=30
    ),
}


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
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    train = synthetic_loader(
        args.train_size, args.batch_size, seed=1, learnable=True, noise=args.noise
    )
    test = synthetic_loader(
        args.eval_size, args.batch_size, seed=2, learnable=True, noise=args.noise
    )
    arm_dir = workdir / name
    arm_dir.mkdir(parents=True, exist_ok=True)
    trainer = DeepSpeechTrainer(
        model,
        CTCLoss(reduction="sum", zero_infinity=True),
        optimizer,
        epochs=args.max_epochs,
        model_path=str(arm_dir / "best.pth"),
        output_file=str(arm_dir / "output.txt"),
        device=args.device,
        device_test=args.device,
        mixed_precision=True,
        stop_cer=args.target,
        seed=args.seed,
        **ARMS[name],
    )
    t0 = time.perf_counter()
    metrics = trainer.run(train, test)
    total = time.perf_counter() - t0
    return {
        "arm": name,
        "reached": metrics.time_to_target_s is not None,
        "time_to_target_s": None
        if metrics.time_to_target_s is None
        else round(metrics.time_to_target_s, 2),
        "total_s": round(total, 2),
        "epochs_run": len(trainer.epoch_seconds),
        "stopped_at_epoch": metrics.stopped_at,
        "best_cer": metrics.best_cer,
        "mean_epoch_s": round(sum(trainer.epoch_seconds) / max(1, len(trainer.epoch_seconds)), 3),
    }


def markdown(results: List[Dict[str, Any]], meta: Dict[str, Any]) -> str:
    base = next((r for r in results if r["arm"] == "legacy"), results[0])
    lines = [
        f"Device `{meta['device']}` · torch {meta['torch']} · target CER <= {meta['target']} · "
        f"learnable synthetic task · GRU {meta['layers']}x{meta['hidden']}",
        "",
        "| arm | time to target (s) | speed-up vs legacy | epochs | mean epoch (s) | best CER |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in results:
        t = r["time_to_target_s"]
        sp = f"{base['time_to_target_s'] / t:.2f}x" if t and base["time_to_target_s"] else "n/a"
        lines.append(
            f"| {r['arm']} | {t if t is not None else 'not reached'} | {sp} | {r['epochs_run']} "
            f"| {r['mean_epoch_s']} | {r['best_cer']:.2f} |"
        )
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--target", type=float, default=10.0, help="stop at this CER (percent)")
    p.add_argument("--max-epochs", type=int, default=60)
    p.add_argument("--device", default="auto")
    p.add_argument("--train-size", type=int, default=1024)
    p.add_argument("--eval-size", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--noise", type=float, default=0.5, help="std of the additive noise")
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
    meta = dict(
        device=str(resolve_device(args.device)),
        torch=torch.__version__,
        platform=platform.platform(),
        target=args.target,
        layers=args.layers,
        hidden=args.hidden,
        train_size=args.train_size,
        eval_size=args.eval_size,
    )
    print("\n" + markdown(results, meta))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"meta": meta, "results": results}, indent=2))


if __name__ == "__main__":
    main()
