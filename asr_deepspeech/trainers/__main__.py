import torch
from torch.nn import CTCLoss
from torch.optim.lr_scheduler import StepLR
from asr_deepspeech.modules import DeepSpeech
from asr_deepspeech.trainers import DeepSpeechTrainer
from sakura import asr_metrics
from asr_deepspeech import cfg
from sakura.ml import AsyncTrainer

if __name__ == "__main__":

    # Instantiate the model, optimizer and scheduler
    model = DeepSpeech(**vars(cfg.model))

    # Init the loaders
    (train_loader, _), (test_loader, _) = model.get_loader(
        manifest=cfg.loaders.train_manifest,
        batch_size=cfg.loaders.batch_size,
        num_workers=cfg.loaders.num_workers,
        caching=cfg.loaders.caching,
    ), model.get_loader(
        manifest=cfg.loaders.val_manifest,
        batch_size=cfg.loaders.batch_size,
        num_workers=cfg.loaders.num_workers,
        caching=cfg.loaders.caching,
    )

    optimizer = torch.optim.AdamW(
        params=model.parameters(),
        lr=cfg.optim.lr,
        betas=eval(cfg.optim.betas),
        eps=cfg.optim.eps,
        weight_decay=cfg.optim.weight_decay,
    )
    scheduler = StepLR(optimizer, step_size=cfg.optim.step, gamma=cfg.optim.gamma)

    # Instantiate the trainer
    trainer = DeepSpeechTrainer(
        model=model,
        criterion=CTCLoss(reduction="sum"),
        optimizer=optimizer,
        scheduler=scheduler,
        metrics=asr_metrics,
        **vars(cfg.trainer)
    )
    trainer = AsyncTrainer(trainer=trainer)

    # Run the trainer
    trainer.run(train_loader=train_loader, test_loader=test_loader)
