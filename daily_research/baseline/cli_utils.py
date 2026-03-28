from __future__ import annotations

from pathlib import Path
from typing import Callable


def parse_stock_list(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [stock.strip().upper() for stock in str(raw).split(",") if stock.strip()]


def load_stock_list_from_file(path: str | None) -> list[str]:
    if not path:
        return []
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"stocks file not found: {path}")

    text = file_path.read_text(encoding="utf-8-sig").strip()
    if not text:
        return []

    tokens: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if "," in line:
            tokens.extend(item.strip() for item in line.split(",") if item.strip())
        else:
            tokens.append(line)
    return [token.upper() for token in tokens]


def parse_csv_list(
    raw: str | None,
    *,
    normalizer: Callable[[str], str] | None = str.lower,
) -> list[str]:
    if not raw:
        return []
    values = [item.strip() for item in str(raw).split(",") if item.strip()]
    if normalizer is None:
        return values
    return [normalizer(item) for item in values]


def parse_int_tuple(raw: str | None, fallback: int) -> tuple[int, ...]:
    if not raw:
        return (int(fallback),)
    values = tuple(int(item.strip()) for item in str(raw).split(",") if item.strip())
    return values or (int(fallback),)


def parse_horizon_weights(raw: str | None) -> dict[int, float]:
    if not raw:
        return {}
    out: dict[int, float] = {}
    for item in str(raw).split(","):
        item = item.strip()
        if not item:
            continue
        horizon_raw, weight_raw = item.split(":", 1)
        out[int(horizon_raw.strip())] = float(weight_raw.strip())
    return out


def parse_state_horizon_profiles(raw: str | None) -> dict[str, dict[int, float]]:
    if not raw:
        return {}
    out: dict[str, dict[int, float]] = {}
    for chunk in str(raw).split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        state_raw, weights_raw = chunk.split("=", 1)
        out[state_raw.strip()] = parse_horizon_weights(weights_raw)
    return out


def parse_state_ensemble_weights(raw: str | None) -> dict[str, dict[str, float]]:
    if not raw:
        return {}
    out: dict[str, dict[str, float]] = {}
    for chunk in str(raw).split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        state_raw, weights_raw = chunk.split("=", 1)
        weights: dict[str, float] = {}
        for item in weights_raw.split(","):
            item = item.strip()
            if not item:
                continue
            name_raw, value_raw = item.split(":", 1)
            weights[name_raw.strip().lower()] = float(value_raw.strip())
        out[state_raw.strip()] = weights
    return out


def parse_named_windows(raw: str | None) -> list[tuple[str, str, str]]:
    if not raw:
        return []
    windows: list[tuple[str, str, str]] = []
    for chunk in str(raw).split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts = chunk.split(":")
        if len(parts) != 3:
            raise ValueError(f"Invalid window spec: {chunk}")
        name, start, end = parts
        windows.append((name.strip(), start.strip(), end.strip()))
    return windows
