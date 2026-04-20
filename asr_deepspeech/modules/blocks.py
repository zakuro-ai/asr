import torch
import torch.nn as nn
import torch.nn.functional as F


class SequenceWise(nn.Module):
    def __init__(self, module):
        """Collapses input of dim T*N*H to (T*N)*H, applies module, reshapes back."""
        super().__init__()
        self.module = module

    def forward(self, x):
        t, n = x.size(0), x.size(1)
        x = x.view(t * n, -1)
        x = self.module(x)
        x = x.view(t, n, -1)
        return x

    def __repr__(self):
        return self.__class__.__name__ + "(\n" + self.module.__repr__() + "\n)"


class MaskConv(nn.Module):
    def __init__(self, seq_module):
        """
        Applies a sequential conv stack while zeroing out padded positions.
        Input shape: (B, C, D, T).
        """
        super().__init__()
        self.seq_module = seq_module

    def forward(self, x, lengths):
        for module in self.seq_module:
            x = module(x)
            mask = torch.zeros(x.size(), dtype=torch.bool, device=x.device)
            for i, length in enumerate(lengths):
                length = length.item()
                if mask[i].size(2) - length > 0:
                    mask[i].narrow(2, length, mask[i].size(2) - length).fill_(True)
            x = x.masked_fill(mask, 0)
        return x, lengths


class InferenceBatchSoftmax(nn.Module):
    def forward(self, input_):
        if not self.training:
            return F.softmax(input_, dim=-1)
        return input_


class BatchRNN(nn.Module):
    def __init__(self, input_size, hidden_size, rnn_type=nn.LSTM, bidirectional=False, batch_norm=True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self._bidirectional = bidirectional
        self.batch_norm = SequenceWise(nn.BatchNorm1d(input_size)) if batch_norm else None
        self.rnn = rnn_type(
            input_size=input_size,
            hidden_size=hidden_size,
            bidirectional=bidirectional,
            bias=True,
        )
        self.num_directions = 2 if bidirectional else 1

    def flatten_parameters(self):
        self.rnn.flatten_parameters()

    def forward(self, x, output_lengths):
        if self.batch_norm is not None:
            x = self.batch_norm(x)
        # pack_padded_sequence requires lengths on CPU
        x = nn.utils.rnn.pack_padded_sequence(x, output_lengths.cpu())
        x, _ = self.rnn(x)
        x, _ = nn.utils.rnn.pad_packed_sequence(x)
        if self._bidirectional:
            # (T x N x H*2) → (T x N x H) by summing directions
            x = x.view(x.size(0), x.size(1), 2, -1).sum(2).view(x.size(0), x.size(1), -1)
        return x


class Lookahead(nn.Module):
    """Wang et al. 2016 — Lookahead Convolution Layer for Unidirectional RNNs.
    Input/output shape: T x N x H.
    """

    def __init__(self, n_features, context):
        super().__init__()
        assert context > 0
        self.context = context
        self.n_features = n_features
        self.pad = (0, self.context - 1)
        self.conv = nn.Conv1d(
            n_features,
            n_features,
            kernel_size=context,
            stride=1,
            groups=n_features,
            padding=0,
            bias=False,
        )

    def forward(self, x):
        x = x.transpose(0, 1).transpose(1, 2)
        x = F.pad(x, pad=self.pad, value=0)
        x = self.conv(x)
        x = x.transpose(1, 2).transpose(0, 1).contiguous()
        return x

    def __repr__(self):
        return f"{self.__class__.__name__}(n_features={self.n_features}, context={self.context})"
