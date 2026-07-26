from __future__ import annotations

import sys
from contextlib import contextmanager
from shutil import get_terminal_size
from typing import Iterator

_STDOUT_AVAILABLE = True
_PROGRESS_SUPPRESSION_DEPTH = 0


def _write_stdout(text: str) -> bool:
    global _STDOUT_AVAILABLE
    if _PROGRESS_SUPPRESSION_DEPTH:
        return True
    if not _STDOUT_AVAILABLE:
        return False
    try:
        sys.stdout.write(text)
        sys.stdout.flush()
        return True
    except OSError as exc:
        if getattr(exc, "errno", None) in {5, 22, 32}:
            _STDOUT_AVAILABLE = False
            return False
        raise
    except ValueError:
        _STDOUT_AVAILABLE = False
        return False


@contextmanager
def suppress_progress() -> Iterator[None]:
    """Temporarily silence progress chatter while preserving result stdout."""

    global _PROGRESS_SUPPRESSION_DEPTH
    _PROGRESS_SUPPRESSION_DEPTH += 1
    try:
        yield
    finally:
        _PROGRESS_SUPPRESSION_DEPTH = max(0, _PROGRESS_SUPPRESSION_DEPTH - 1)


def _terminal_width() -> int:
    return max(get_terminal_size((120, 20)).columns, 60)


def _truncate_text(text: str, max_width: int) -> str:
    clean = str(text or "").strip()
    if max_width <= 0:
        return ""
    if len(clean) <= max_width:
        return clean
    if max_width <= 3:
        return clean[:max_width]
    return clean[: max_width - 3] + "..."


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
        self.desc = str(desc or "Working")
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
        self.desc = str(desc or "Working")
        self.refresh()

    def update(self, n: int = 1) -> None:
        self.n = min(self.total, self.n + max(int(n), 0))
        self.refresh()

    def refresh(self) -> None:
        if self.closed:
            return
        ratio = 1.0 if self.total <= 0 else min(max(self.n / self.total, 0.0), 1.0)
        percent_text = f"{ratio * 100:5.1f}%"
        term_width = _terminal_width()
        bar_width = max(16, min(28, term_width // 5))
        filled = int(round(bar_width * ratio))
        bar = "#" * filled + "-" * max(bar_width - filled, 0)
        counts = f"{self.n}/{self.total} {self.unit}".strip()
        prefix = f"\r[{bar}] {percent_text} "
        available_text = max(term_width - len(prefix) - len(counts) - 3, 10)
        body = _truncate_text(self.desc, available_text)
        line = f"{prefix}{body} | {counts}"
        padding = max(self._last_width - len(line), 0)
        if _write_stdout(line + (" " * padding)):
            self._last_width = len(line)

    def close(self) -> None:
        if self.closed:
            return
        if self.leave:
            self.refresh()
            _write_stdout("\n")
        else:
            _write_stdout("\r" + (" " * self._last_width) + "\r")
        self.closed = True


def create_progress(
    total: int,
    desc: str,
    unit: str = "step",
    leave: bool = True,
    position: int = 0,
):
    return _SimpleProgress(total=total, desc=desc, unit=unit, leave=leave, position=position)


def progress_write(message: str) -> None:
    _write_stdout(f"{message}\n")
