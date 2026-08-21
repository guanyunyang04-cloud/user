"""Auxiliary Tail Update: config responsibilities."""

from __future__ import annotations

from quantlab.data.qdp_v2 import normalization as _normalization

_normalize_cninfo_dividend = _normalization.normalize_cninfo_dividend


_normalize_cninfo_share_change = _normalization.normalize_cninfo_share_change


class AuxiliaryTailUpdateError(RuntimeError):
    pass
