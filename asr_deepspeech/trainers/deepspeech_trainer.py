from tqdm import tqdm
import torch.utils.data.distributed
from asr_deepspeech import check_loss
import os
from sakura.ml import SakuraTrainer
from gnutools.fs import parent
from sakura.ml.decorators import parallel


class DeepSpeechTrainer(SakuraTrainer):
    def __init__(self,
                 model,
                 criterion,
                 epochs,
                 metrics,
                 optimizer,
                 scheduler,
                 model_path,
                 checkpoint_path,
                 device,
                 device_test):
        super(DeepSpeechTrainer, self).__init__(model,
                                                optimizer=optimizer,
                                                scheduler=scheduler,
                                                metrics=metrics,
                                                epochs=epochs,
                                                model_path=model_path,
                                                checkpoint_path=checkpoint_path,
                                                device=device,
                                                device_test=device_test)
        self.criterion = criterion
        self.load()


    def run(self, train_loader, test_loader):
        for self._epoch in self._epochs:
            self.train(train_loader)
            self.test(test_loader)
            self.checkpoint()

    def checkpoint(self):
        if self._metrics.test.current == self._metrics.test.best:
            os.makedirs(parent(self._model_path), exist_ok=True)
            torch.save(self._model.state_dict(), self._model_path)

    def description(self):
        lr = self._optimizer.param_groups[0]["lr"]*pow(10, 5)
        current, best = self._metrics.test.current, self._metrics.test.best
        suffix = f" | CER: {current.cer:.4f} / ({best.cer:.4f})"
        suffix += f" | Loss:{current.loss:.4f} / ({best.loss:.4f})"
        return f"({self._epochs.best}) {self._model.id}{suffix} | Lr: {lr:.4f}e-5 | Epoch: {self._epochs.current}/{self._epochs.total}"

    @parallel
    def train(self, train_loader):
        scaler = torch.cuda.amp.GradScaler()
        self._model.train()
        self._model.to(self._device)
        self.optimizer_to(self._optimizer, self._device)
        current, best = self._metrics.train.current, self._metrics.train.best
        loader = train_loader
        for iter, data in tqdm(enumerate(loader, start=0),
                               total=len(loader),
                               desc=self.description()):
            inputs, targets, input_percentages, target_sizes = data
            input_sizes = input_percentages.mul_(int(inputs.size(3))).int()
            with torch.cuda.amp.autocast():
                # measure data loading time
                inputs = inputs.to(self._device)
                out, output_sizes = self._model.forward(inputs, input_sizes)
                out = out.transpose(0, 1)  # TxNxH
                float_out = out.float()  # ensure float32 for loss
                float_out = float_out.log_softmax(2)
                loss = self.criterion(float_out, targets, output_sizes, target_sizes).to(self._device)
                loss = loss / inputs.size(0)  # average the loss by minibatch

                # Check the loss
                loss_value = loss.item()
                valid_loss, error = check_loss(loss, loss_value)
                if valid_loss:
                    self._optimizer.zero_grad()
                    scaler.scale(loss).backward()
                    # loss.backward()
                    current.loss+=loss_value
                    scaler.step(self._optimizer)
                    scaler.update()
                    # self._optimizer.step()
                else:
                    print("Loss non valid, skipped")
                    pass
        self.update(current, best, loader)
        self._scheduler.step()

    @parallel
    def test(self, test_loader):
        current, best = self._metrics.test.current, self._metrics.test.best
        loader = test_loader
        wer, cer, _ = self._model(loader=loader, device=self._device_test)
        current.wer, current.cer = wer, cer
        self.update(current, best, loader)

    def update(self, current, best, loader):
        current.loss /= len(loader.dataset)
        try:
            assert best.cer is not None
            assert best.cer < current.cer
        except AssertionError:
            vars(best).update(vars(current))
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
        ckpt = torch.load("model.pth", map_location='cpu')
        if all:
            self._optimizer.load_state_dict(ckpt["optimizer"].state_dict())
            self._scheduler = ckpt["scheduler"]
            self._scheduler.optimizer = self._optimizer
        self._metrics = ckpt["metrics"]
        self._epochs.start, self._epochs.current, self._epochs.best = ckpt["epoch"], ckpt["epoch"], ckpt["epoch"]


    def save(self, file_path):
        torch.save({
            "epoch": self._epochs.best,
            "metrics": self._metrics,
            "optimizer": self._optimizer,
            "scheduler": self._scheduler,
            "state_dict": self._model.state_dict()
        }, file_path)

