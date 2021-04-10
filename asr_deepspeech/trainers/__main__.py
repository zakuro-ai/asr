from torch.nn import CTCLoss
from asr_deepspeech.modules import DeepSpeech
from asr_deepspeech.trainers import DeepSpeechTrainer
from asr_deepspeech import load_config
import torch
from torch.optim.lr_scheduler import StepLR

from sakura.ml import AsyncTrainer
from sakura import asr_metrics

if __name__ == '__main__':
    config = load_config()

    model = DeepSpeech(**vars(config.model))

    optimizer = torch.optim.SGD(model.parameters(),
                                lr=config.optim.lr,
                                momentum=config.optim.momentum,
                                nesterov=config.optim.nesterov,
                                weight_decay=config.optim.weight_decay)

    scheduler = StepLR(optimizer,
                       step_size=config.optim.step,
                       gamma=config.optim.gamma)

    trainer = DeepSpeechTrainer(model=model,
                                criterion=CTCLoss(reduction="sum"),
                                optimizer=optimizer,
                                scheduler=scheduler,
                                metrics=asr_metrics,
                                **vars(config.trainer))

    trainer = AsyncTrainer(trainer, device_test="cpu")

    train_loader, train_sampler = model.get_loader(manifest=config.loaders.train_manifest,
                                                   batch_size=config.loaders.batch_size,
                                                   num_workers=config.loaders.num_workers)

    test_loader, test_sampler = model.get_loader(manifest=config.loaders.val_manifest,
                                                 batch_size=config.loaders.batch_size,
                                                 num_workers=config.loaders.num_workers)
    trainer.run(train_loader, test_loader)

