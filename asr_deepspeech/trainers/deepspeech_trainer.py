from tqdm import tqdm
import torch.utils.data.distributed
from asr_deepspeech import check_loss
from argparse import Namespace
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
                 scheduler):
        super(DeepSpeechTrainer, self).__init__(model,
                                                optimizer=optimizer,
                                                scheduler=scheduler,
                                                metrics=metrics,
                                                epochs=epochs,
                                                model_path=os.path.realpath("test.pth"),
                                                checkpoint_path=os.path.realpath("test.ckpt.pth"))
        self.criterion = criterion

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
        current, best = self._metrics.test.current, self._metrics.test.best
        suffix = f" | CER: {current.cer:.4f} / ({best.cer:.4f})"
        suffix += f" | Loss:{current.loss:.4f} / ({best.loss:.4f})"
        return f"({self._epochs.best}) ASR | Epoch: {self._epochs.current}/{self._epochs.total}{suffix}"

    @parallel
    def train(self, train_loader):
        self._model.train()
        self._model.cuda()
        current, best = self._metrics.test.current, self._metrics.test.best
        loader = train_loader
        for iter, data in tqdm(enumerate(loader, start=0),
                               total=len(loader),
                               desc=self.description()):
            inputs, targets, input_percentages, target_sizes = data
            input_sizes = input_percentages.mul_(int(inputs.size(3))).int()
            # measure data loading time
            inputs = inputs.to('cuda')
            out, output_sizes = self._model.forward(inputs, input_sizes)
            out = out.transpose(0, 1)  # TxNxH
            float_out = out.float()  # ensure float32 for loss
            float_out = float_out.log_softmax(2)
            loss = self.criterion(float_out, targets, output_sizes, target_sizes).to('cuda')
            loss = loss / inputs.size(0)  # average the loss by minibatch

            # Check the loss
            loss_value = loss.item()
            valid_loss, error = check_loss(loss, loss_value)
            if valid_loss:
                self._optimizer.zero_grad()
                loss.backward()
                current.loss+=loss_value
                self._optimizer.step()
            else:
                print("Loss non valid, skipped")
                pass
        self.update(current, best, loader)

    @parallel
    def test(self, test_loader):
        current, best = self._metrics.test.current, self._metrics.test.best
        loader = test_loader
        wer, cer, _ = self._model(loader=loader, device="cpu")
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

    def load(self):
        try:
            assert self.continue_from is not None
            assert os.path.exists(self.continue_from)
            ckpt = Namespace(**torch.load(self.continue_from))
            self.metrics = ckpt.metrics
            self.optim = ckpt.optim
            self.optimizer = ckpt.optimizer
            self.model.load_state_dict(ckpt.state_dict)
            self.update(save=False)
            self.start_epoch = self.lu+1
        except:
            pass

    # def save(self, file_path):
    #     torch.save({
    #         "metrics": self.metrics,
    #         "optim": self.optim,
    #         "optimizer": self.optimizer,
    #         "state_dict": self.model.state_dict()
    #     }, file_path)
    #
