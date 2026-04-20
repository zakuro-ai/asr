import os

import torch
import torch.nn as nn
from tqdm import tqdm

from asr_deepspeech import check_loss
from gnutools.fs import parent
from sakura.ml import SakuraTrainer


class DeepSpeechTrainer(SakuraTrainer):
    def __init__(
        self,
        model,
        criterion,
        epochs,
        metrics,
        optimizer,
        model_path,
        checkpoint_path,
        device,
        device_test,
        mixed_precision,
        output_file,
        scheduler=None,
        overwrite_lr=None,
        max_norm: float = 400.0,
        compile_model: bool = False,
    ):
        super().__init__(
            model,
            optimizer=optimizer,
            scheduler=scheduler,
            metrics=metrics,
            epochs=epochs,
            model_path=model_path,
            checkpoint_path=checkpoint_path,
            device=device,
            device_test=device_test,
        )
        self.criterion = criterion
        self.mixed_precision = mixed_precision
        self.output_file = output_file
        self.overwrite_lr = overwrite_lr
        self.max_norm = max_norm

        self._use_amp = mixed_precision and str(device) != "cpu"
        # GradScaler is created once so it accumulates loss-scale history across epochs.
        self._scaler = torch.amp.GradScaler("cuda") if self._use_amp else None

        if compile_model:
            self._model = torch.compile(self._model)

        self.load()

    def run(self, train_loader, test_loader):
        for self._epoch in self._epochs:
            self.train(train_loader)
            self.test(test_loader)

    def checkpoint(self):
        if self._metrics.test.current == self._metrics.test.best:
            self.save()

    def description(self):
        lr = self._optimizer.param_groups[0]["lr"] * 1e5
        cur, best = self._metrics.test.current, self._metrics.test.best
        tcur, tbest = self._metrics.train.current, self._metrics.train.best
        cer_str = f"{cur.cer:.4f}/({best.cer:.4f})" if cur.cer is not None else "n/a"
        return (
            f"({self._epochs.best}) {self._model.id}"
            f" | CER: {cer_str}"
            f" | Loss: {tcur.loss:.4f}/({tbest.loss:.4f})"
            f" | Lr: {lr:.4f}e-5"
            f" | Epoch: {self._epochs.current}/{self._epochs.total}"
        )

    def train(self, train_loader):
        self._model.train()
        self._model.to(self._device)
        self.optimizer_to(self._optimizer, self._device)
        current, best = self._metrics.train.current, self._metrics.train.best

        for data in tqdm(train_loader, total=len(train_loader), desc=self.description()):
            if self._use_amp:
                with torch.amp.autocast("cuda"):
                    valid_loss, loss, loss_value = self.fit(data)
            else:
                valid_loss, loss, loss_value = self.fit(data)

            if valid_loss:
                self._optimizer.zero_grad()
                if self._use_amp:
                    self._scaler.scale(loss).backward()
                    self._scaler.unscale_(self._optimizer)
                    nn.utils.clip_grad_norm_(self._model.parameters(), self.max_norm)
                    self._scaler.step(self._optimizer)
                    self._scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(self._model.parameters(), self.max_norm)
                    self._optimizer.step()
                current.loss += loss_value
            else:
                print("Loss non-valid, skipped")

        self.update(current, best, train_loader, update_best=False)
        if self._scheduler is not None:
            self._scheduler.step()

    def fit(self, data):
        inputs, targets, input_percentages, target_sizes = data
        input_sizes = input_percentages.mul_(int(inputs.size(3))).int()
        inputs = inputs.to(self._device)
        out, output_sizes = self._model.forward(inputs, input_sizes)
        out = out.transpose(0, 1)
        float_out = out.float().log_softmax(2)
        loss = self.criterion(float_out, targets, output_sizes, target_sizes).to(self._device)
        loss = loss / inputs.size(0)
        loss_value = loss.item()
        valid_loss, _ = check_loss(loss, loss_value)
        return valid_loss, loss, loss_value

    def test(self, test_loader):
        current, best = self._metrics.test.current, self._metrics.test.best
        wer, cer, _ = self._model(
            loader=test_loader, device=self._device_test, output_file=self.output_file
        )
        current.wer, current.cer = wer, cer
        self.update(current, best, test_loader, update_best=True)
        self.checkpoint()

    def update(self, current, best, loader, update_best=False):
        current.loss /= len(loader.dataset)
        try:
            assert best.cer is not None
            assert best.cer < current.cer
        except AssertionError:
            vars(best).update(vars(current))
            if update_best:
                self._epochs.best = self._epochs.current

    @staticmethod
    def optimizer_to(optim, device):
        for param in optim.state.values():
            if isinstance(param, torch.Tensor):
                param.data = param.data.to(device)
                if param._grad is not None:
                    param._grad.data = param._grad.data.to(device)
            elif isinstance(param, dict):
                for subparam in param.values():
                    if isinstance(subparam, torch.Tensor):
                        subparam.data = subparam.data.to(device)
                        if subparam._grad is not None:
                            subparam._grad.data = subparam._grad.data.to(device)

    def load(self, all=True):
        if os.path.exists(self._model_path):
            ckpt = torch.load(self._model_path, map_location="cpu", weights_only=False)
            self._model.load_state_dict(ckpt["state_dict"])
            if all:
                self._optimizer.load_state_dict(ckpt["optimizer"])
                if self.overwrite_lr is not None:
                    self._optimizer.param_groups[0]["lr"] = self.overwrite_lr
                self._scheduler = ckpt.get("scheduler", self._scheduler)
                if self._scheduler is not None:
                    self._scheduler.optimizer = self._optimizer
                if ckpt.get("scaler") is not None and self._scaler is not None:
                    self._scaler.load_state_dict(ckpt["scaler"])
            self._metrics = ckpt["metrics"]
            epoch = ckpt["epoch"]
            self._epochs.start = self._epochs.current = self._epochs.best = epoch
            print(f"restart from {self._model_path}")

    def save(self):
        os.makedirs(parent(self._model_path), exist_ok=True)
        torch.save(
            {
                "epoch": self._epochs.best,
                "metrics": self._metrics,
                "optimizer": self._optimizer.state_dict(),
                "scheduler": self._scheduler,
                "scaler": self._scaler.state_dict() if self._scaler is not None else None,
                "state_dict": self._model.state_dict(),
            },
            self._model_path,
        )
        print(f"{self._model_path} saved")
