from __future__ import annotations

import sys
from contextlib import contextmanager
from shutil import get_terminal_size
from typing import Iterator, TypeVar

try:
    from tqdm.auto import tqdm as _tqdm
except Exception:  # pragma: no cover - optional dependency
    _tqdm = None


T = TypeVar("T")


def _stage_text(label: str, current: int, total: int, message: str, detail: str = "") -> str:
    text = f"{label} {int(current)}/{int(total)} {message}"
    detail = str(detail or "").strip()
    if detail:
        return f"{text} | {detail}"
    return text


class _SimpleProgress:
    def __init__(
        self,
        total: int,
        desc: str,
        unit: str = "step",
        leave: bool = True,
        position: int = 0,
    ) -> None:
        self.total = max(int(total), 0)
        self.desc = str(desc)
        self.unit = str(unit or "step")
        self.leave = bool(leave)
        self.position = int(position)
        self.n = 0
        self.closed = False
        self._last_width = 0
        self.refresh()

    def __enter__(self) -> "_SimpleProgress":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def set_description_str(self, desc: str) -> None:
        self.desc = str(desc)
        self.refresh()

    def update(self, n: int = 1) -> None:
        self.n = min(self.total, self.n + max(int(n), 0))
        self.refresh()

    def refresh(self) -> None:
        if self.closed:
            return
        ratio = 1.0 if self.total <= 0 else min(max(self.n / self.total, 0.0), 1.0)
        prefix = f"{self.desc} "
        suffix = f" {self.n}/{self.total} {self.unit}"
        term_width = max(get_terminal_size((100, 20)).columns, 40)
        bar_width = max(12, min(36, term_width - len(prefix) - len(suffix) - 8))
        filled = int(round(bar_width * ratio))
        bar = "#" * filled + "-" * max(bar_width - filled, 0)
        line = f"\r{prefix}[{bar}]{suffix}"
        padding = max(self._last_width - len(line), 0)
        sys.stdout.write(line + (" " * padding))
        sys.stdout.flush()
        self._last_width = len(line)

    def close(self) -> None:
        if self.closed:
            return
        if self.leave:
            self.refresh()
            sys.stdout.write("\n")
        else:
            sys.stdout.write("\r" + (" " * self._last_width) + "\r")
        sys.stdout.flush()
        self.closed = True


def create_progress(
    total: int,
    desc: str,
    unit: str = "step",
    leave: bool = True,
    position: int = 0,
):
    if _tqdm is not None:
        return _tqdm(
            total=total,
            desc=desc,
            unit=unit,
            leave=leave,
            position=position,
            dynamic_ncols=True,
            ascii=True,
        )
    return _SimpleProgress(total=total, desc=desc, unit=unit, leave=leave, position=position)


def iter_progress(
    iterable,
    *,
    total: int | None,
    desc: str,
    unit: str = "item",
    leave: bool = False,
    position: int = 0,
):
    if _tqdm is not None:
        return _tqdm(
            iterable,
            total=total,
            desc=desc,
            unit=unit,
            leave=leave,
            position=position,
            dynamic_ncols=True,
            ascii=True,
        )

    def _generator() -> Iterator[T]:
        resolved_total = int(total or 0)
        with create_progress(
            total=resolved_total,
            desc=desc,
            unit=unit,
            leave=leave,
            position=position,
        ) as progress:
            for item in iterable:
                yield item
                progress.update(1)

    return _generator()


def progress_write(message: str) -> None:
    if _tqdm is not None:
        _tqdm.write(str(message))
    else:
        print(str(message))


class StageProgress:
    def __init__(self, total: int, label: str, *, position: int = 0) -> None:
        self.total = max(int(total), 1)
        self.label = str(label or "流程")
        self.position = int(position)
        self.current = 0
        self._progress = None

    def __enter__(self) -> "StageProgress":
        self._progress = create_progress(
            total=self.total,
            desc=_stage_text(self.label, 0, self.total, "准备中"),
            unit="step",
            leave=True,
            position=self.position,
        )
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._progress is None:
            return
        if exc_type is None and self.current >= self.total:
            self._progress.set_description_str(_stage_text(self.label, self.total, self.total, "完成"))
        self._progress.close()

    @contextmanager
    def stage(self, message: str, detail: str = "") -> Iterator[None]:
        if self._progress is None:
            raise RuntimeError("StageProgress must be entered before creating stages.")
        step_no = min(self.current + 1, self.total)
        self._progress.set_description_str(_stage_text(self.label, step_no, self.total, message, detail))
        try:
            yield
        except Exception:
            progress_write(_stage_text(self.label, step_no, self.total, f"失败: {message}", detail))
            raise
        else:
            self._progress.update(step_no - self.current)
            self.current = step_no

    def log(self, message: str) -> None:
        progress_write(message)
