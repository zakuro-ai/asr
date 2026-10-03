"""DeepSpeech2 trainer, accelerated by the Sakura runtime.

The trainer owns the training loop. With ``runtime="sakura"`` (default) it drives a
:class:`sakura.SakuraRuntime` through a :class:`sakura.adapters.DDPAdapter` and installs

* ``MixedPrecision`` -- autocast + GradScaler handled by the runtime (CUDA only),
* ``AsyncEval``      -- per-epoch WER/CER evaluation overlapped with the next epoch of
                        training; an adaptive gate measures both modes and keeps the
                        faster one, so it can never be slower than synchronous eval,
* ``AsyncCheckpoint``-- rolling, atomic, resumable checkpoints written off the training
                        thread.

With ``runtime="vanilla"`` the exact same loop runs with synchronous evaluation and a
hand-rolled GradScaler. It is the reference used by ``benchmarks/run.py``.
"""

from __future__ import annotations

import logging
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from asr_deepspeech.checkpoint import (
    build_state,
    load_checkpoint,
    save_checkpoint,
    snapshot,
)
from asr_deepspeech.device import autocast, make_grad_scaler, resolve_device
from asr_deepspeech.functional import check_loss
from asr_deepspeech.metrics import EvalResult, Metrics

log = logging.getLogger("asr_deepspeech")

RUNTIMES = ("sakura", "vanilla")


def seed_everything(seed: Optional[int]) -> None:
    """Seed python, numpy and torch (no-op when ``seed`` is None)."""
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(
    model: nn.Module, loader: Any, device: torch.device, output_file: Optional[str] = None
) -> EvalResult:
    """Run ``model`` over ``loader`` and return WER / CER in percent."""
    wer, cer, _ = model(loader=loader, device=device, output_file=output_file)
    return EvalResult(wer=float(wer), cer=float(cer))


class DeepSpeechTrainer:
    """Train and evaluate a :class:`~asr_deepspeech.modules.DeepSpeech` model.

    Args:
        model: the network. It must be callable on a loader to evaluate (see
            ``DeepSpeech.__call__``).
        criterion: CTC loss (``reduction="sum"``; the trainer divides by batch size).
        optimizer: optimizer over ``model.parameters()``.
        scheduler: optional LR scheduler, stepped once per epoch.
        epochs: total number of epochs (a resumed run continues up to this number).
        model_path: where the best-CER model is written (also a resume source).
        output_file: optional text report written by every evaluation.
        checkpoint_path: directory for rolling resume checkpoints. Defaults to
            ``<model_path dir>/checkpoints``.
        device, device_test: ``"auto"``/``"cuda"``/``"cpu"`` for training / evaluation
            (``device_test`` is where the asynchronous evaluation replica lives).
        mixed_precision: enable autocast on CUDA.
        overwrite_lr: override the learning rate after restoring a checkpoint.
        runtime: ``"sakura"`` (accelerated) or ``"vanilla"`` (synchronous reference).
        async_eval: overlap evaluation with training (``runtime="sakura"`` only).
        rolling_checkpoints: keep the two newest per-epoch resume checkpoints (written
            asynchronously with ``runtime="sakura"``, synchronously otherwise).
        seed: optional RNG seed.
    """

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any] = None,
        *,
        epochs: int,
        model_path: str,
        output_file: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
        device: Any = "auto",
        device_test: Any = "auto",
        mixed_precision: bool = True,
        overwrite_lr: Optional[float] = None,
        runtime: str = "sakura",
        async_eval: bool = True,
        rolling_checkpoints: bool = True,
        seed: Optional[int] = None,
    ) -> None:
        if runtime not in RUNTIMES:
            raise ValueError(f"runtime must be one of {RUNTIMES}, got {runtime!r}")
        if epochs < 1:
            raise ValueError("epochs must be >= 1")
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.epochs = int(epochs)
        self.model_path = str(model_path)
        self.output_file = output_file
        self.checkpoint_dir = str(checkpoint_path or Path(model_path).parent / "checkpoints")
        self.device = resolve_device(device)
        self.device_test = resolve_device(device_test)
        self.mixed_precision = bool(mixed_precision)
        self.overwrite_lr = overwrite_lr
        self.runtime = runtime
        self.async_eval = bool(async_eval)
        self.rolling_checkpoints = bool(rolling_checkpoints)
        self.seed = seed

        self.metrics = Metrics()
        self.start_epoch = 0
        self.epoch_seconds: List[float] = []
        self._epoch = -1
        self._states: Dict[int, Dict[str, torch.Tensor]] = {}
        self._folded = 0
        self._async_eval_svc: Any = None

    # ------------------------------------------------------------------ public

    def run(self, train_loader: Any, test_loader: Any) -> Metrics:
        """Train for the remaining epochs and return the run :class:`Metrics`."""
        seed_everything(self.seed)
        self._resume()
        if self.start_epoch >= self.epochs:
            log.info("nothing to do: checkpoint is already at epoch %d", self.start_epoch)
            return self.metrics
        if self.runtime == "sakura":
            self._run_sakura(train_loader, test_loader)
        else:
            self._run_vanilla(train_loader, test_loader)
        return self.metrics

    # --------------------------------------------------------------- run modes

    def _run_vanilla(self, train_loader: Any, test_loader: Any) -> None:
        scaler = make_grad_scaler(self.device, enabled=self.mixed_precision)
        use_amp = self.mixed_precision and self.device.type == "cuda"

        def step(loss: torch.Tensor) -> None:
            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()
            else:
                loss.backward()
                self.optimizer.step()

        for epoch in range(self.start_epoch, self.epochs):
            t0 = time.perf_counter()
            self._epoch = epoch
            train_loss = self._train_epoch(
                train_loader, epoch, backward_step=step, use_autocast=use_amp
            )
            self._scheduler_step()
            self.metrics.record(epoch, train_loss=train_loss)
            result = evaluate(self.model, test_loader, self.device_test, self.output_file)
            self._on_eval(epoch, result, state=self._live_state())
            if self.rolling_checkpoints:
                self._write_last(epoch)
            self.epoch_seconds.append(time.perf_counter() - t0)

    def _run_sakura(self, train_loader: Any, test_loader: Any) -> None:
        from sakura import SakuraRuntime
        from sakura.adapters import DDPAdapter
        from sakura.dispatch import ThreadDispatcher
        from sakura.services import AsyncCheckpoint, AsyncEval, MixedPrecision

        self._async_eval_svc = None
        self._folded = 0
        replica = self._make_replica() if self.async_eval else None
        payload = {"replica": replica, "loader": test_loader}

        def eval_fn(epoch: int, payload: Dict[str, Any]) -> Dict[str, Any]:
            payload["replica"].load_state_dict(self._states[epoch])
            r = evaluate(payload["replica"], payload["loader"], self.device_test, self.output_file)
            return {"wer": r.wer, "cer": r.cer}

        def sync_eval_fn(model: nn.Module, loader: Any) -> Dict[str, Any]:
            r = evaluate(model, loader, self.device_test, self.output_file)
            return {"wer": r.wer, "cer": r.cer}

        eval_dispatcher, ckpt_dispatcher = (
            ThreadDispatcher(max_workers=1),
            ThreadDispatcher(max_workers=1),
        )
        try:
            with SakuraRuntime(record_history=False) as rt:
                if self.mixed_precision and self.device.type == "cuda":
                    rt.install(MixedPrecision(dtype="auto"))
                if self.async_eval:
                    self._async_eval_svc = AsyncEval(
                        eval_fn=eval_fn,
                        eval_payload=payload,
                        dispatcher=eval_dispatcher,
                        sync_eval_fn=sync_eval_fn,
                        adaptive=True,
                        total_epochs=self.epochs,
                        max_pending=1,
                        on_backpressure="block",
                    )
                    rt.install(self._async_eval_svc)
                if self.rolling_checkpoints:
                    rt.install(
                        AsyncCheckpoint(
                            dir=self.checkpoint_dir,
                            dispatcher=ckpt_dispatcher,
                            state_provider=lambda: snapshot(self._full_state(self._epoch)),
                            every="epoch",
                            keep=2,
                            writer=lambda state, path: {"path": save_checkpoint(state, path)},
                        )
                    )
                adapter = DDPAdapter(rt, rank=0, world_size=1)
                self._train_with_adapter(rt, adapter, train_loader, test_loader)
                self._fold_results()
        finally:
            eval_dispatcher.shutdown()
            ckpt_dispatcher.shutdown()

    def _train_with_adapter(
        self, rt: Any, adapter: Any, train_loader: Any, test_loader: Any
    ) -> None:
        self.model.to(self.device)
        adapter.on_train_begin(self.model, self.optimizer, train_loader, test_loader)

        def step(loss: torch.Tensor) -> None:
            rt.scale_loss(loss).backward()
            adapter.on_optimizer_step(self.optimizer)
            if not rt.optimizer_step(self.optimizer):
                self.optimizer.step()

        for epoch in range(self.start_epoch, self.epochs):
            t0 = time.perf_counter()
            self._epoch = epoch
            adapter.on_epoch_begin(epoch)
            train_loss = self._train_epoch(
                train_loader, epoch, backward_step=step, on_batch=adapter.on_train_step_begin
            )
            self._scheduler_step()
            self.metrics.record(epoch, train_loss=train_loss)
            svc = self._async_eval_svc
            if svc is not None and svc.wants_snapshot():
                self._states[epoch] = self._live_state()
            if svc is None:  # synchronous evaluation, like the vanilla runtime
                result = evaluate(self.model, test_loader, self.device_test, self.output_file)
                self._on_eval(epoch, result, state=self._live_state())
            adapter.on_epoch_end(epoch, self.model, self.optimizer, {"train_loss": train_loss})
            self._fold_results()
            self.epoch_seconds.append(time.perf_counter() - t0)
        adapter.on_train_end(self.model)

    # ---------------------------------------------------------------- training

    def _train_epoch(
        self,
        loader: Any,
        epoch: int,
        *,
        backward_step: Any,
        use_autocast: bool = False,
        on_batch: Any = None,
    ) -> float:
        """One pass over ``loader``; returns the mean loss over the valid batches."""
        self.model.train()
        self.model.to(self.device)
        _optimizer_to(self.optimizer, self.device)
        total, valid = 0.0, 0
        desc = f"epoch {epoch + 1}/{self.epochs}"
        for step, batch in enumerate(tqdm(loader, desc=desc, leave=False)):
            if on_batch is not None:
                on_batch(self.model, batch, step)
            self.optimizer.zero_grad(set_to_none=True)
            with autocast(self.device, enabled=use_autocast):
                loss = self._forward_loss(batch)
            ok, error = check_loss(loss, loss.item())
            if not ok:
                self.metrics.skipped_batches += 1
                log.warning("epoch %d step %d: skipped batch (%s)", epoch, step, error)
                continue
            backward_step(loss)
            total += loss.item()
            valid += 1
        if valid == 0:
            raise RuntimeError(f"epoch {epoch}: every batch had an invalid loss; aborting")
        return total / valid

    def _forward_loss(self, batch: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        inputs, targets, input_percentages, target_sizes = batch
        input_sizes = input_percentages.mul(int(inputs.size(3))).int()
        out, output_sizes = self.model.forward(inputs.to(self.device), input_sizes)
        log_probs = out.transpose(0, 1).float().log_softmax(2)  # T x N x H, fp32 for CTC
        loss = self.criterion(log_probs, targets, output_sizes, target_sizes)
        return loss.to(self.device) / inputs.size(0)

    def _scheduler_step(self) -> None:
        if self.scheduler is not None:
            self.scheduler.step()

    # ------------------------------------------------------------- evaluation

    def _make_replica(self) -> nn.Module:
        import copy

        replica = copy.deepcopy(self.model).to(self.device_test)
        replica.eval()
        return replica

    def _fold_results(self) -> None:
        """Merge resolved asynchronous evaluations into :attr:`metrics`."""
        if self._async_eval_svc is None:
            return
        history = self._async_eval_svc.history
        for rec in history[self._folded :]:
            epoch = rec["epoch"]
            if "cer" in rec:
                state = self._states.pop(epoch, None)
                if state is None and epoch == self._epoch:
                    state = self._live_state()
                self._on_eval(epoch, EvalResult(wer=rec["wer"], cer=rec["cer"]), state=state)
            else:
                self._states.pop(epoch, None)
                log.warning("epoch %d: evaluation skipped (%s)", epoch, rec.get("reason"))
        self._folded = len(history)

    def _on_eval(self, epoch: int, result: EvalResult, state: Optional[Dict[str, Any]]) -> None:
        self.metrics.record(epoch, wer=result.wer, cer=result.cer)
        if self.metrics.update_best(epoch, result.cer):
            log.info("epoch %d: new best CER %.2f (WER %.2f)", epoch, result.cer, result.wer)
            if state is not None:
                payload = {
                    "format": 2,
                    "epoch": epoch,
                    "state_dict": state,
                    "optimizer": None,
                    "scheduler": None,
                    "metrics": self.metrics.state_dict(),
                }
                save_checkpoint(payload, self.model_path)

    # ------------------------------------------------------------ checkpoints

    def _live_state(self) -> Dict[str, torch.Tensor]:
        return {k: v.detach().to("cpu", copy=True) for k, v in self.model.state_dict().items()}

    def _full_state(self, epoch: int) -> Dict[str, Any]:
        return build_state(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            epoch=epoch,
            metrics=self.metrics.state_dict(),
        )

    def _write_last(self, epoch: int) -> None:
        save_checkpoint(
            snapshot(self._full_state(epoch)), Path(self.checkpoint_dir) / f"epoch_{epoch:04d}.pt"
        )
        for old in sorted(Path(self.checkpoint_dir).glob("epoch_*.pt"))[:-2]:
            old.unlink(missing_ok=True)

    def _resume(self) -> None:
        """Restore from the newest rolling checkpoint, else from ``model_path``."""
        rolling = sorted(Path(self.checkpoint_dir).glob("epoch_*.pt"))
        candidates = [str(rolling[-1])] if rolling else []
        candidates.append(self.model_path)
        for path in candidates:
            if Path(path).exists():
                self._restore(load_checkpoint(path), path)
                return

    def _restore(self, ckpt: Dict[str, Any], path: str) -> None:
        self.model.load_state_dict(ckpt["state_dict"])
        if ckpt.get("optimizer") is not None:
            self.optimizer.load_state_dict(ckpt["optimizer"])
            if self.overwrite_lr is not None:
                for group in self.optimizer.param_groups:
                    group["lr"] = self.overwrite_lr
        if ckpt.get("scheduler") is not None and self.scheduler is not None:
            self.scheduler.load_state_dict(ckpt["scheduler"])
        self.metrics = Metrics.from_state_dict(ckpt.get("metrics") or {})
        self.start_epoch = int(ckpt["epoch"]) + 1
        log.info("restored %s (epoch %d)", path, ckpt["epoch"])


def _optimizer_to(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    """Move optimizer state tensors to ``device`` (needed after a CPU-side restore)."""
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)
