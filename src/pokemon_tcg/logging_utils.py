"""One logger interface, swappable backend.

TensorBoard today (`TensorBoardLogger`); a Weights & Biases backend can be
added later behind the same `RunLogger` interface without touching any
call site -- no `wandb` import exists anywhere in this module today.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol


class RunLogger(Protocol):
    def log_scalar(self, tag: str, value: float, step: int) -> None: ...
    def log_scalars(self, tags_values: dict[str, float], step: int) -> None: ...
    def close(self) -> None: ...


class TensorBoardLogger:
    def __init__(self, log_dir: Path) -> None:
        from torch.utils.tensorboard import SummaryWriter

        log_dir.mkdir(parents=True, exist_ok=True)
        self._writer = SummaryWriter(log_dir=str(log_dir))

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        self._writer.add_scalar(tag, value, step)

    def log_scalars(self, tags_values: dict[str, float], step: int) -> None:
        for tag, value in tags_values.items():
            self._writer.add_scalar(tag, value, step)

    def close(self) -> None:
        self._writer.close()
