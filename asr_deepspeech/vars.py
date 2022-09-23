import torch
from torch import nn
import scipy.signal

N =-1
windows =\
    {
        'hamming': scipy.signal.hamming,
        'hann': scipy.signal.hann,
        'blackman': scipy.signal.blackman,
        'bartlett': scipy.signal.bartlett
    }

torch.manual_seed(123456)
torch.cuda.manual_seed_all(123456)

supported_rnns = {
    'lstm': nn.LSTM,
    'rnn': nn.RNN,
    'gru': nn.GRU
}
supported_rnns_inv = dict((v, k) for k, v in supported_rnns.items())

