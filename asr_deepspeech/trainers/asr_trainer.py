# import numpy as np
# from datetime import timedelta
# import math
# from .default_trainer import DefaultTrainer
# from asr_deepspeech import AverageMeter


# class ASRTrainer(DefaultTrainer):
#     def __init__(self, model, args):
#         super(ASRTrainer, self).__init__(model, args)
#         self.labels = self.net.labels
#         self.batch_time = AverageMeter()
#         self.data_time = AverageMeter()
#         # Copy the attributes from the model
#         self.avg_loss = model._avg_loss
#         self.loss_results = model._loss_results
#         self.wer_results = model._wer_results
#         self.cer_results = model._cer_results
#         self.start_epoch = model._start_epoch
#         self.start_iter = model._start_iter
#         self.iter = model._start_iter
#         self.epoch = model._start_epoch
#         self.lu = model._start_epoch
#         self.wer = model._wer
#         self.cer = model._cer
#         self.best_wer = model._best_wer
#         self.best_cer = model._best_cer
#         self.optimizer = model._optimizer
#         self.audio_conf = self.net.audio_conf
#         self.checkpoint_per_batch = args.checkpoint_per_batch

#     def description(self):
#         __description__ = " | ".join(
#             [
#                 f"{self.manifest} - {timedelta(seconds=int(self.epoch_time))} >> {self.epoch + 1}/{self.epochs} ({self.lu + 1})",
#                 f'Lr {self.optimizer.param_groups[0]["lr"] * 1000:.2f}e-3',
#                 f"Loss {self.average_loss():.4f}"
#                 if self.average_loss() is not None
#                 else "",
#                 f"WER/CER {self.wer:.2f}/{self.cer:.2f} - ({self.best_wer:.2f}/[{self.best_cer:.2f}])"
#                 if self.best_cer is not None
#                 else "",
#             ]
#         )
#         return __description__

#     def average_loss(self):
#         try:
#             avg_loss = np.mean(np.array(self.loss_results)[: self.epoch])
#             assert not math.isnan(avg_loss)
#             return avg_loss
#         except AssertionError:
#             pass
