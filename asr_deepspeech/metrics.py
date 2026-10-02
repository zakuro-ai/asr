"""Training / evaluation bookkeeping for the DeepSpeech trainer."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class EvalResult:
    """Word / character error rates (in percent) for one evaluation pass."""

    wer: float
    cer: float


@dataclass
class Metrics:
    """Running state of a training run.

    ``history`` holds one record per epoch (``epoch``, ``train_loss`` and, once the
    evaluation for that epoch has resolved, ``wer`` / ``cer``). Evaluations may resolve
    late when they are overlapped with training, so records are merged by epoch.
    """

    best_cer: Optional[float] = None
    best_epoch: Optional[int] = None
    skipped_batches: int = 0
    history: List[Dict[str, Any]] = field(default_factory=list)

    def record(self, epoch: int, **values: Any) -> Dict[str, Any]:
        """Merge ``values`` into the history record of ``epoch`` (created on demand)."""
        for rec in self.history:
            if rec["epoch"] == epoch:
                rec.update(values)
                return rec
        rec = {"epoch": epoch, **values}
        self.history.append(rec)
        return rec

    def update_best(self, epoch: int, cer: float) -> bool:
        """Register ``cer`` for ``epoch``; return True if it is a new best."""
        if self.best_cer is None or cer < self.best_cer:
            self.best_cer, self.best_epoch = cer, epoch
            return True
        return False

    def state_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_state_dict(cls, state: Dict[str, Any]) -> "Metrics":
        return cls(**{k: v for k, v in state.items() if k in cls.__dataclass_fields__})
