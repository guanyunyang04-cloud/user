from __future__ import annotations

import sys
from contextlib import contextmanager
from shutil import get_terminal_size
from typing import Iterator, TypeVar


T = TypeVar("T")

_ACTIVE_STAGE_PROGRESS_STACK: list["StageProgress"] = []


def _current_stage_progress() -> "StageProgress | None":
    if not _ACTIVE_STAGE_PROGRESS_STACK:
        return None
    return _ACTIVE_STAGE_PROGRESS_STACK[-1]


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
        prefix = f"[{bar}] {percent_text} "
        available_text = max(term_width - len(prefix) - len(counts) - 3, 10)
        body = _truncate_text(self.desc, available_text)
        line = f"\r{prefix}{body} | {counts}"
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


class _StageChildProgress:
    def __init__(
        self,
        owner: "StageProgress",
        total: int,
        desc: str,
        unit: str = "step",
        leave: bool = True,
        position: int = 0,
    ) -> None:
        self.owner = owner
        self.total = max(int(total), 0)
        self.desc = str(desc or "Working")
        self.unit = str(unit or "step")
        self.leave = bool(leave)
        self.position = int(position)
        self.n = 0
        self.closed = False
        self.owner._attach_child(self)

    def __enter__(self) -> "_StageChildProgress":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def set_description_str(self, desc: str) -> None:
        self.desc = str(desc or "Working")
        self.owner._refresh()

    def update(self, n: int = 1) -> None:
        self.n = min(self.total, self.n + max(int(n), 0))
        self.owner._refresh()

    def refresh(self) -> None:
        self.owner._refresh()

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        self.owner._detach_child(self)


def create_progress(
    total: int,
    desc: str,
    unit: str = "step",
    leave: bool = True,
    position: int = 0,
):
    active_stage = _current_stage_progress()
    if active_stage is not None:
        return _StageChildProgress(
            owner=active_stage,
            total=total,
            desc=desc,
            unit=unit,
            leave=leave,
            position=position,
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
    active_stage = _current_stage_progress()
    if active_stage is not None:
        active_stage.log(str(message))
        return
    print(str(message))


class StageProgress:
    _SPINNER_FRAMES = "|/-\\"

    def __init__(self, total: int, label: str, *, position: int = 0, leave: bool = True) -> None:
        self.total = max(int(total), 1)
        self.label = str(label or "Workflow")
        self.position = int(position)
        self.leave = bool(leave)
        self.current = 0
        self.current_stage_no = 1
        self.current_message = "Preparing"
        self.current_detail = ""
        self.note = ""
        self._child_stack: list[_StageChildProgress] = []
        self._primary_child: _StageChildProgress | None = None
        self._closed = False
        self._last_width = 0
        self._refresh_tick = 0

    def __enter__(self) -> "StageProgress":
        _ACTIVE_STAGE_PROGRESS_STACK.append(self)
        self._refresh()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is None:
            self.complete()
        self.close()

    def _compose_line(self) -> str:
        child = self._child_stack[-1] if self._child_stack else None
        primary_child_ratio = 0.0
        if self._primary_child is not None and self._primary_child.total > 0:
            primary_child_ratio = min(
                max(self._primary_child.n / self._primary_child.total, 0.0),
                1.0,
            )
        ratio = min(max((self.current + primary_child_ratio) / self.total, 0.0), 1.0)
        percent_text = f"{ratio * 100:5.1f}%"
        term_width = _terminal_width()
        bar_width = max(18, min(30, term_width // 5))
        filled = int(round(bar_width * ratio))
        bar = "#" * filled + "-" * max(bar_width - filled, 0)
        spinner = (
            self._SPINNER_FRAMES[self._refresh_tick % len(self._SPINNER_FRAMES)]
            if child is not None
            else "-"
        )
        stage_prefix = (
            f"{self.label} {min(max(self.current_stage_no, 1), self.total)}/{self.total} "
            f"{self.current_message}"
        )
        parts = [stage_prefix]
        if self.current_detail:
            parts.append(self.current_detail)
        if child is not None:
            child_counts = f"{child.n}/{child.total} {child.unit}".strip()
            parts.append(f"{child.desc} [{child_counts}]")
        if self.note:
            parts.append(self.note)
        body = " | ".join(part for part in parts if str(part).strip())
        prefix = f"[{bar}] {percent_text} {spinner} "
        available_text = max(term_width - len(prefix) - 1, 16)
        return f"\r{prefix}{_truncate_text(body, available_text)}"

    def _refresh(self) -> None:
        if self._closed:
            return
        line = self._compose_line()
        padding = max(self._last_width - len(line), 0)
        sys.stdout.write(line + (" " * padding))
        sys.stdout.flush()
        self._last_width = len(line)
        self._refresh_tick += 1

    def _attach_child(self, child: _StageChildProgress) -> None:
        is_root_child = not self._child_stack
        if child in self._child_stack:
            self._child_stack = [item for item in self._child_stack if item is not child]
        self._child_stack.append(child)
        if is_root_child and self._primary_child is None:
            self._primary_child = child
        self._refresh()

    def _detach_child(self, child: _StageChildProgress) -> None:
        self._child_stack = [item for item in self._child_stack if item is not child]
        self._refresh()

    def start_stage(self, stage_no: int, message: str, detail: str = "") -> None:
        stage_no = max(1, min(int(stage_no), self.total))
        if stage_no > 1:
            self.current = max(self.current, stage_no - 1)
        self.current_stage_no = stage_no
        self.current_message = str(message or "Working")
        self.current_detail = str(detail or "").strip()
        self.note = ""
        self._child_stack = []
        self._primary_child = None
        self._refresh()

    def complete_stage(self, stage_no: int | None = None) -> None:
        if stage_no is None:
            self.current = max(self.current, min(self.current_stage_no, self.total))
        else:
            self.current = max(self.current, min(int(stage_no), self.total))
        self._child_stack = []
        self._primary_child = None
        self._refresh()

    def complete(self) -> None:
        self.current = self.total
        self.current_stage_no = self.total
        self.current_message = "Done"
        self.current_detail = ""
        self.note = ""
        self._child_stack = []
        self._primary_child = None
        self._refresh()

    def close(self) -> None:
        if self._closed:
            return
        if self.leave:
            self._refresh()
            sys.stdout.write("\n")
        else:
            sys.stdout.write("\r" + (" " * self._last_width) + "\r")
        sys.stdout.flush()
        self._closed = True
        if self in _ACTIVE_STAGE_PROGRESS_STACK:
            _ACTIVE_STAGE_PROGRESS_STACK[:] = [
                item for item in _ACTIVE_STAGE_PROGRESS_STACK if item is not self
            ]

    @contextmanager
    def stage(self, message: str, detail: str = "") -> Iterator[None]:
        step_no = min(self.current + 1, self.total)
        self.start_stage(step_no, message, detail)
        try:
            yield
        except Exception:
            self.log(f"Failed: {message}")
            raise
        else:
            self.complete_stage(step_no)

    def log(self, message: str) -> None:
        self.note = str(message or "").strip()
        self._refresh()
