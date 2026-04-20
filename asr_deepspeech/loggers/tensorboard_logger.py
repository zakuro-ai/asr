import os
from torch.utils.tensorboard import SummaryWriter
from asr_deepspeech import to_np


class TensorBoardLogger:
    def __init__(self, id, log_dir, log_params):
        os.makedirs(log_dir, exist_ok=True)
        self.id = id
        self.writer = SummaryWriter(log_dir)
        self.log_params = log_params

    def update(self, epoch, values, parameters=None):
        loss = values["loss_results"][epoch + 1]
        wer = values["wer_results"][epoch + 1]
        cer = values["cer_results"][epoch + 1]
        self.writer.add_scalars(
            self.id,
            {"Avg Train Loss": loss, "Avg WER": wer, "Avg CER": cer},
            epoch + 1,
        )
        if self.log_params and parameters is not None:
            for tag, value in parameters():
                tag = tag.replace(".", "/")
                self.writer.add_histogram(tag, to_np(value), epoch + 1)
                self.writer.add_histogram(tag + "/grad", to_np(value.grad), epoch + 1)

    def load_previous_values(self, start_epoch, values):
        for i in range(start_epoch):
            self.writer.add_scalars(
                self.id,
                {
                    "Avg Train Loss": values["loss_results"][i],
                    "Avg WER": values["wer_results"][i],
                    "Avg CER": values["cer_results"][i],
                },
                i + 1,
            )
