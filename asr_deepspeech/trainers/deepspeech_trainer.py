from tqdm import tqdm
import torch.utils.data.distributed
from asr_deepspeech import check_loss
import os
from sakura.ml import SakuraTrainer
from gnutools.fs import parent


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
    ):
        super(DeepSpeechTrainer, self).__init__(
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
        self.load()

    def run(self, train_loader, test_loader):
        for self._epoch in self._epochs:
            self.train(train_loader)
            self.test(test_loader)

    def checkpoint(self):
        if self._metrics.test.current == self._metrics.test.best:
            self.save()

    def description(self):
        lr = self._optimizer.param_groups[0]["lr"] * pow(10, 5)
        current, best = self._metrics.test.current, self._metrics.test.best
        tcurrent, tbest = self._metrics.train.current, self._metrics.train.best
        suffix = f" | CER: {current.cer:.4f} / ({best.cer:.4f})"
        suffix += f" | Loss:{tcurrent.loss:.4f} / ({tbest.loss:.4f})"
        return f"({self._epochs.best}) {self._model.id}{suffix} | Lr: {lr:.4f}e-5 | Epoch: {self._epochs.current}/{self._epochs.total}"

    def train(self, train_loader):
        self._model.train()
        self._model.to(self._device)
        self.optimizer_to(self._optimizer, self._device)
        current, best = self._metrics.train.current, self._metrics.train.best
        loader = train_loader
        scaler = torch.cuda.amp.GradScaler() if self.mixed_precision else None

        for iter, data in tqdm(
            enumerate(loader, start=0), total=len(loader), desc=self.description()
        ):

            if self.mixed_precision:
                with torch.cuda.amp.autocast():
                    valid_loss, loss, loss_value = self.fit(data)
            else:
                valid_loss, loss, loss_value = self.fit(data)

            if valid_loss:
                self._optimizer.zero_grad()
                if self.mixed_precision:
                    scaler.scale(loss).backward()
                    scaler.step(self._optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    self._optimizer.step()
                current.loss += loss_value
            else:
                print("Loss non valid, skipped")
                pass

        self.update(current, best, loader, update_best=False)
        self._scheduler.step() if self._scheduler is not None else None

    def fit(self, data):
        inputs, targets, input_percentages, target_sizes = data
        input_sizes = input_percentages.mul_(int(inputs.size(3))).int()
        # measure data loading time
        inputs = inputs.to(self._device)
        out, output_sizes = self._model.forward(inputs, input_sizes)
        out = out.transpose(0, 1)  # TxNxH
        float_out = out.float()  # ensure float32 for loss
        float_out = float_out.log_softmax(2)
        loss = self.criterion(float_out, targets, output_sizes, target_sizes).to(
            self._device
        )
        loss = loss / inputs.size(0)  # average the loss by minibatch

        # Check the loss
        loss_value = loss.item()
        valid_loss, error = check_loss(loss, loss_value)
        return valid_loss, loss, loss_value

    def test(self, test_loader):
        current, best = self._metrics.test.current, self._metrics.test.best
        loader = test_loader
        wer, cer, _ = self._model(
            loader=loader, device=self._device_test, output_file=self.output_file
        )
        current.wer, current.cer = wer, cer
        self.update(current, best, loader, update_best=True)
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
            # Not sure there are any global tensors in the state dict
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
        model_path = self._model_path
        if os.path.exists(model_path):
            ckpt = torch.load(model_path, map_location="cpu")
            self._model.load_state_dict(ckpt["state_dict"])
            if all:
                self._optimizer.load_state_dict(ckpt["optimizer"])
                if self.overwrite_lr is not None:
                    self._optimizer.param_groups[0]["lr"] = self.overwrite_lr
                self._scheduler = (
                    ckpt["scheduler"]
                    if ckpt["scheduler"] is not None
                    else self._scheduler
                )
                if self._scheduler is not None:
                    self._scheduler.optimizer = self._optimizer
            self._metrics = ckpt["metrics"]
            self._epochs.start, self._epochs.current, self._epochs.best = (
                ckpt["epoch"],
                ckpt["epoch"],
                ckpt["epoch"],
            )
            print(f"restart from {model_path}")

    def save(self):
        os.makedirs(parent(self._model_path), exist_ok=True)
        torch.save(
            {
                "epoch": self._epochs.best,
                "metrics": self._metrics,
                "optimizer": self._optimizer.state_dict(),
                "scheduler": self._scheduler,
                "state_dict": self._model.state_dict(),
            },
            self._model_path,
        )
        print(f"{self._model_path} saved...")
