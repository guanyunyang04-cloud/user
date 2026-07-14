from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd

from quant_data_platform.core.paths import qdp_paths


CORRECTIONS_SCHEMA_VERSION = "qdp_v3_corrections_v1"
CORRECTION_FIELDS = frozenset(
    {
        "correction_id",
        "provider",
        "domain",
        "key",
        "field",
        "old_value",
        "new_value",
        "effective_from",
        "effective_to",
        "reason",
    }
)
DATE_COLUMNS = ("trade_date", "divid_operate_date", "event_date", "date")


def default_corrections_config(workspace_root: str | Path | None = None) -> Path:
    return qdp_paths(workspace_root).project_root / "configs" / "qdp_v3_corrections.json"


def _date(value: Any, *, field: str) -> str:
    text = str(value or "").strip()[:10]
    try:
        parsed = pd.Timestamp(text)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"correction_invalid_date:{field}:{value}") from exc
    if parsed.strftime("%Y-%m-%d") != text:
        raise ValueError(f"correction_invalid_date:{field}:{value}")
    return text


def _canonical_key(key: Mapping[str, Any]) -> str:
    return json.dumps(dict(key), ensure_ascii=True, sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True)
class CorrectionRule:
    correction_id: str
    provider: str
    domain: str
    key: dict[str, Any]
    field: str
    old_value: Any
    new_value: Any
    effective_from: str
    effective_to: str
    reason: str

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "CorrectionRule":
        fields = set(payload)
        if fields != CORRECTION_FIELDS:
            raise ValueError(
                f"correction_fields_invalid:missing={sorted(CORRECTION_FIELDS - fields)}:extra={sorted(fields - CORRECTION_FIELDS)}"
            )
        key = payload.get("key")
        if not isinstance(key, Mapping) or not key:
            raise ValueError("correction_key_must_be_nonempty_object")
        rule = cls(
            correction_id=str(payload.get("correction_id", "") or "").strip(),
            provider=str(payload.get("provider", "") or "").strip(),
            domain=str(payload.get("domain", "") or "").strip(),
            key={str(name): value for name, value in key.items()},
            field=str(payload.get("field", "") or "").strip(),
            old_value=payload.get("old_value"),
            new_value=payload.get("new_value"),
            effective_from=_date(payload.get("effective_from"), field="effective_from"),
            effective_to=_date(payload.get("effective_to"), field="effective_to"),
            reason=str(payload.get("reason", "") or "").strip(),
        )
        if not all((rule.correction_id, rule.provider, rule.domain, rule.field, rule.reason)):
            raise ValueError(f"correction_required_value_blank:{rule.correction_id or '<unknown>'}")
        if rule.effective_from > rule.effective_to:
            raise ValueError(f"correction_invalid_interval:{rule.correction_id}")
        if _value_matches(rule.old_value, rule.new_value):
            raise ValueError(f"correction_old_and_new_equal:{rule.correction_id}")
        return rule

    def target_identity(self) -> tuple[str, str, str, str]:
        return self.provider, self.domain, _canonical_key(self.key), self.field


@dataclass(frozen=True)
class CorrectionSet:
    rules: tuple[CorrectionRule, ...]
    regressions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        ids = [rule.correction_id for rule in self.rules]
        if len(ids) != len(set(ids)):
            duplicates = sorted({item for item in ids if ids.count(item) > 1})
            raise ValueError(f"correction_id_duplicate:{duplicates}")
        grouped: dict[tuple[str, str, str, str], list[CorrectionRule]] = {}
        for rule in self.rules:
            grouped.setdefault(rule.target_identity(), []).append(rule)
        for target, rules in grouped.items():
            previous_to = ""
            for rule in sorted(rules, key=lambda item: (item.effective_from, item.effective_to)):
                if previous_to and rule.effective_from <= previous_to:
                    raise ValueError(f"correction_interval_overlap:{target}:{rule.correction_id}")
                previous_to = rule.effective_to

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "CorrectionSet":
        if str(payload.get("schema_version", "") or "") != CORRECTIONS_SCHEMA_VERSION:
            raise ValueError("corrections_schema_version_invalid")
        corrections = payload.get("corrections", [])
        regressions = payload.get("regressions", [])
        if not isinstance(corrections, list) or not isinstance(regressions, list):
            raise ValueError("corrections_payload_shape_invalid")
        return cls(
            rules=tuple(CorrectionRule.from_mapping(item) for item in corrections if isinstance(item, Mapping)),
            regressions=tuple(str(item) for item in regressions),
        )


def load_corrections(
    path: str | Path | None = None,
    *,
    workspace_root: str | Path | None = None,
) -> CorrectionSet:
    resolved = Path(path) if path is not None else default_corrections_config(workspace_root)
    if not resolved.exists():
        return CorrectionSet(())
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("corrections_config_must_be_object")
    return CorrectionSet.from_mapping(payload)


def _value_matches(left: Any, right: Any) -> bool:
    if left is None or (not isinstance(left, (list, tuple, dict, set)) and pd.isna(left)):
        return right is None or (not isinstance(right, (list, tuple, dict, set)) and pd.isna(right))
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-12)
    return left == right


def apply_corrections(
    frame: pd.DataFrame,
    *,
    provider: str,
    domain: str,
    corrections: CorrectionSet | Iterable[CorrectionRule],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Return a corrected canonical copy and an inspectable application log."""

    rules = corrections.rules if isinstance(corrections, CorrectionSet) else tuple(corrections)
    selected = [rule for rule in rules if rule.provider == provider and rule.domain == domain]
    if not selected:
        return frame.copy(), []
    out = frame.copy()
    date_column = next((item for item in DATE_COLUMNS if item in out.columns), "")
    if not date_column:
        raise ValueError(f"correction_date_column_missing:{domain}")
    date_values = out[date_column].fillna("").astype(str).str.slice(0, 10)
    applied: list[dict[str, Any]] = []
    for rule in selected:
        if rule.field not in out.columns:
            raise ValueError(f"correction_field_missing:{rule.correction_id}:{rule.field}")
        missing_keys = sorted(set(rule.key) - set(out.columns))
        if missing_keys:
            raise ValueError(f"correction_key_fields_missing:{rule.correction_id}:{missing_keys}")
        mask = date_values.between(rule.effective_from, rule.effective_to)
        for column, expected in rule.key.items():
            mask &= out[column].map(lambda actual: _value_matches(actual, expected))
        indices = out.index[mask]
        if len(indices) == 0:
            raise ValueError(f"correction_target_missing:{rule.correction_id}")
        mismatched = [index for index in indices if not _value_matches(out.at[index, rule.field], rule.old_value)]
        if mismatched:
            raise ValueError(f"correction_old_value_mismatch:{rule.correction_id}:{len(mismatched)}")
        out.loc[indices, rule.field] = rule.new_value
        for index in indices:
            record_key = {column: out.at[index, column] for column in rule.key}
            record_key[date_column] = out.at[index, date_column]
            applied.append(
                {
                    "correction_id": rule.correction_id,
                    "provider": rule.provider,
                    "domain": rule.domain,
                    "record_key": record_key,
                    "field": rule.field,
                    "old_value": rule.old_value,
                    "new_value": rule.new_value,
                    "reason": rule.reason,
                }
            )
    return out, applied
