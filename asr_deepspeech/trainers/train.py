import random

import numpy as np
import torch
from torch.nn import CTCLoss
from torch.optim.lr_scheduler import StepLR

from asr_deepspeech import cfg
from asr_deepspeech.metrics import asr_metrics
from asr_deepspeech.modules import DeepSpeech
from asr_deepspeech.trainers import DeepSpeechTrainer


def seed_everything(seed: int = 123456) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main() -> None:
    seed_everything()

    model = DeepSpeech(**vars(cfg.model))

    cache_dir = getattr(cfg.loaders, "cache_dir", None)
    loader_kwargs = dict(
        batch_size=cfg.loaders.batch_size,
        num_workers=cfg.loaders.num_workers,
        cache_dir=cache_dir,
    )
    train_loader, _ = model.get_loader(manifest=cfg.loaders.train_manifest, **loader_kwargs)
    test_loader, _ = model.get_loader(manifest=cfg.loaders.val_manifest, **loader_kwargs)

    optimizer = torch.optim.AdamW(
        params=model.parameters(),
        lr=cfg.optim.lr,
        betas=tuple(cfg.optim.betas),
        eps=cfg.optim.eps,
        weight_decay=cfg.optim.weight_decay,
    )
    scheduler = StepLR(optimizer, step_size=cfg.optim.step, gamma=cfg.optim.gamma)

    trainer = DeepSpeechTrainer(
        model=model,
        criterion=CTCLoss(reduction="sum"),
        optimizer=optimizer,
        scheduler=scheduler,
        metrics=asr_metrics(),
        **vars(cfg.trainer),
    )
    trainer.run(train_loader=train_loader, test_loader=test_loader)


if __name__ == "__main__":
    main()
