"""Checkpoint (de)serialisation.

New checkpoints contain only tensors and plain Python containers, so they can be loaded
with ``torch.load(..., weights_only=True)``. Checkpoints written by asr-deepspeech
<= 0.4 (which pickled ``sakura`` 0.1 metric objects and the scheduler) are still readable
through a tolerant unpickler that replaces unknown classes with plain namespaces.
"""

from __future__ import annotations

import os
import pickle
import tempfile
import types
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch

PathLike = Union[str, "os.PathLike[str]"]
FORMAT_VERSION = 2


class _TolerantUnpickler(pickle.Unpickler):
    """Maps classes that no longer exist (e.g. ``sakura.ml.*``) to ``SimpleNamespace``."""

    def find_class(self, module: str, name: str) -> Any:
        try:
            return super().find_class(module, name)
        except (ImportError, AttributeError):
            return types.SimpleNamespace


_legacy_pickle = types.SimpleNamespace(
    Unpickler=_TolerantUnpickler,
    load=lambda f, **kw: _TolerantUnpickler(f, **kw).load(),
    __name__="legacy_pickle",
)


def build_state(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any],
    epoch: int,
    metrics: Dict[str, Any],
) -> Dict[str, Any]:
    """Assemble the (picklable, ``weights_only``-safe) checkpoint payload."""
    return {
        "format": FORMAT_VERSION,
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "metrics": metrics,
    }


def snapshot(state: Dict[str, Any]) -> Dict[str, Any]:
    """Detach a checkpoint payload onto the CPU so it can be written off-thread."""

    def _cpu(obj: Any) -> Any:
        if isinstance(obj, torch.Tensor):
            return obj.detach().to("cpu", copy=True)
        if isinstance(obj, dict):
            return {k: _cpu(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return type(obj)(_cpu(v) for v in obj)
        return obj

    return _cpu(state)


def save_checkpoint(state: Dict[str, Any], path: PathLike) -> str:
    """Atomically write ``state`` to ``path`` (write to a temp file, then rename)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    os.close(fd)
    try:
        torch.save(state, tmp)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return str(path)


def load_checkpoint(path: PathLike) -> Dict[str, Any]:
    """Load a checkpoint, transparently upgrading legacy (<= 0.4) files.

    Returns a dict with ``epoch``, ``state_dict``, ``optimizer``, ``scheduler`` (a state
    dict or ``None``) and ``metrics`` (a plain dict).
    """
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        ckpt = torch.load(
            path, map_location="cpu", weights_only=False, pickle_module=_legacy_pickle
        )
        ckpt = _upgrade_legacy(ckpt)
    if "state_dict" not in ckpt:
        raise ValueError(f"{path}: not an asr-deepspeech checkpoint (no 'state_dict')")
    return ckpt


def _upgrade_legacy(ckpt: Dict[str, Any]) -> Dict[str, Any]:
    scheduler = ckpt.get("scheduler")
    if scheduler is not None and hasattr(scheduler, "state_dict"):
        scheduler = scheduler.state_dict()
    elif not isinstance(scheduler, dict):
        scheduler = None
    legacy = ckpt.get("metrics")
    test = getattr(getattr(legacy, "test", None), "best", None)
    cer = getattr(test, "cer", None)
    metrics: Dict[str, Any] = {}
    if cer is not None:
        metrics = {"best_cer": float(cer), "best_epoch": int(ckpt.get("epoch", 0))}
    return {
        "format": 1,
        "epoch": int(ckpt.get("epoch", 0)),
        "state_dict": ckpt["state_dict"],
        "optimizer": ckpt.get("optimizer"),
        "scheduler": scheduler,
        "metrics": metrics,
    }
