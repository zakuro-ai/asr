"""Device + mixed-precision resolution, centralized for CPU/GPU portability."""

import warnings

import torch


def resolve_device(spec="auto") -> torch.device:
    """Resolve a device spec to a concrete ``torch.device``.

    Accepts ``"auto"``, ``"cuda"``/``"gpu"``, ``"cpu"``, a ``torch.device``,
    or ``None`` (treated as ``"auto"``). ``"auto"`` selects CUDA when available,
    otherwise CPU. An explicit CUDA request on a machine without a GPU warns and
    falls back to CPU. Unknown specs raise ``ValueError``.
    """
    if spec is None:
        spec = "auto"
    if isinstance(spec, torch.device):
        return spec
    key = str(spec).lower()
    if key == "cpu":
        return torch.device("cpu")
    if key in ("cuda", "gpu"):
        if torch.cuda.is_available():
            return torch.device("cuda")
        warnings.warn(
            "CUDA device requested but no GPU is available; falling back to CPU.",
            UserWarning,
            stacklevel=2,
        )
        return torch.device("cpu")
    if key == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raise ValueError(f"Unknown device spec: {spec!r}")


def make_grad_scaler(device, enabled: bool = True) -> "torch.amp.GradScaler":
    """GradScaler that is enabled only for CUDA (AMP is a no-op elsewhere)."""
    dev = resolve_device(device)
    return torch.amp.GradScaler(dev.type, enabled=enabled and dev.type == "cuda")


def autocast(device, enabled: bool = True):
    """Autocast context; a no-op unless running on CUDA with ``enabled``."""
    dev = resolve_device(device)
    return torch.amp.autocast(device_type=dev.type, enabled=enabled and dev.type == "cuda")
