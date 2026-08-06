from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).resolve().parents[1] / "seq100_hot_money_path_hits.py"
SPEC = importlib.util.spec_from_file_location("hot_money_path_hits", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_first_hit_respects_invalid_tail() -> None:
    mask = np.array([[False, True, True], [False, False, True]])
    valid = np.array([[True, True, True], [True, False, False]])
    result = MODULE._first_hit(mask, valid)
    assert result[0] == 2.0
    assert np.isnan(result[1])


def test_first_hit_can_report_legal_t_plus_one_path_days() -> None:
    mask = np.array([[True, False, False]])
    valid = np.ones_like(mask, dtype=bool)
    result = MODULE._first_hit(mask, valid, np.array([2.0, 3.0, 4.0]))
    assert result[0] == 2.0


def test_first_hit_missing_is_nan() -> None:
    result = MODULE._first_hit(np.zeros((2, 3), dtype=bool), np.ones((2, 3), dtype=bool))
    assert np.isnan(result).all()
