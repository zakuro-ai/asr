import torch

class VisdomLogger(object):
    def __init__(self, id, num_epochs):
        from visdom import Visdom
        self.viz = Visdom()
        self.opts = dict(title=id, ylabel='', xlabel='Epoch', legend=['Loss', 'WER', 'CER'])
        self.viz_window = None
        self.epochs = torch.arange(1, num_epochs + 1)
        self.visdom_plotter = True

    def update(self, epoch, values):
        x_axis = self.epochs[0:epoch + 1]
        y_axis = torch.stack((values["loss_results"][:epoch + 1],
                              values["wer_results"][:epoch + 1],
                              values["cer_results"][:epoch + 1]),
                             dim=1)
        self.viz_window = self.viz.line(
            X=x_axis,
            Y=y_axis,
            opts=self.opts,
            win=self.viz_window,
            update='replace' if self.viz_window else None
        )

    def load_previous_values(self, start_epoch, package):
        self.update(start_epoch - 1, package)  # Add all values except the iteration we're starting from
