"""Train DeepSpeech2 from the packaged (or ``$ZAK_ASR_CONFIG``) configuration.

python -m asr_deepspeech.trainers [--runtime {sakura,vanilla}] [--no-async-eval]
"""

from __future__ import annotations

import argparse
import ast
import logging
from typing import Any, Sequence, Tuple

import torch
from torch.nn import CTCLoss
from torch.optim.lr_scheduler import StepLR

from asr_deepspeech import cfg
from asr_deepspeech.modules import DeepSpeech
from asr_deepspeech.trainers import DeepSpeechTrainer


def parse_betas(value: Any) -> Tuple[float, ...]:
    """Parse optimizer betas from a string like ``"(0.9, 0.999)"`` or a native list."""
    if isinstance(value, (tuple, list)):
        return tuple(float(x) for x in value)
    return tuple(float(x) for x in ast.literal_eval(value))


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="asr_deepspeech.trainers", description=__doc__)
    parser.add_argument("--runtime", choices=("sakura", "vanilla"), help="override trainer.runtime")
    parser.add_argument(
        "--no-async-eval", action="store_true", help="evaluate synchronously (sakura runtime)"
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    trainer_cfg = dict(vars(cfg.trainer))
    if args.runtime:
        trainer_cfg["runtime"] = args.runtime
    if args.no_async_eval:
        trainer_cfg["async_eval"] = False

    model = DeepSpeech(**vars(cfg.model))
    loader_args = dict(
        batch_size=cfg.loaders.batch_size,
        num_workers=cfg.loaders.num_workers,
        caching=cfg.loaders.caching,
    )
    train_loader, _ = model.get_loader(manifest=cfg.loaders.train_manifest, **loader_args)
    test_loader, _ = model.get_loader(manifest=cfg.loaders.val_manifest, **loader_args)

    optimizer = torch.optim.AdamW(
        params=model.parameters(),
        lr=cfg.optim.lr,
        betas=parse_betas(cfg.optim.betas),
        eps=cfg.optim.eps,
        weight_decay=cfg.optim.weight_decay,
    )
    scheduler = StepLR(optimizer, step_size=cfg.optim.step, gamma=cfg.optim.gamma)

    trainer = DeepSpeechTrainer(
        model=model,
        criterion=CTCLoss(reduction="sum"),
        optimizer=optimizer,
        scheduler=scheduler,
        **trainer_cfg,
    )
    trainer.run(train_loader, test_loader)


if __name__ == "__main__":
    main()
