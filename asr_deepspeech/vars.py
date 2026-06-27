import scipy.signal.windows as _windows
import torch
from torch import nn

N = -1
windows = {
    "hamming": _windows.hamming,
    "hann": _windows.hann,
    "blackman": _windows.blackman,
    "bartlett": _windows.bartlett,
}

torch.manual_seed(123456)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(123456)

supported_rnns = {"lstm": nn.LSTM, "rnn": nn.RNN, "gru": nn.GRU}
supported_rnns_inv = dict((v, k) for k, v in supported_rnns.items())


def resolve_rnn_type(spec):
    """Map an RNN spec to its class without ``eval``.

    Accepts ``"nn.LSTM"``, ``"LSTM"``, ``"lstm"`` (and rnn/gru variants) or an
    already-resolved class (passthrough).
    """
    if isinstance(spec, type):
        return spec
    key = str(spec).split(".")[-1].lower()
    try:
        return supported_rnns[key]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported rnn_type {spec!r}; expected one of {sorted(supported_rnns)}"
        ) from exc
