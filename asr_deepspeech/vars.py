from torch import nn
import scipy.signal.windows as _windows

windows = {
    'hamming': _windows.hamming,
    'hann': _windows.hann,
    'blackman': _windows.blackman,
    'bartlett': _windows.bartlett,
}

supported_rnns = {
    'lstm': nn.LSTM,
    'rnn': nn.RNN,
    'gru': nn.GRU,
}
supported_rnns_inv = {v: k for k, v in supported_rnns.items()}
