from __future__ import annotations

PATH_POLICY_DEFAULT_MAINLINE = "seq100_todayclose_path_only"
PATH_POLICY_DEFAULT_PROFILE = "seq100_todayclose_path_only_daily_only_summary_v2"
SEQ100_MAINLINE_ID = PATH_POLICY_DEFAULT_MAINLINE
SEQ100_MAINLINE_PROFILE = PATH_POLICY_DEFAULT_PROFILE

# Archived lineage constants. Kept explicit so old evidence code remains named as history.
ALPHA_PATH20_POLICY_VERSION = "alpha_path20_neural_policy_v1"
ALPHA_PATH20_POLICY_PROFILE = "alpha_path20_neural_policy_v1"
ALPHA_PATH20_SEQUENCE_POLICY_VERSION = "alpha_path20_sequence_policy_v1"
ALPHA_PATH20_SEQUENCE_POLICY_PROFILE = "alpha_path20_sequence_policy_v1"

__all__ = [
    "PATH_POLICY_DEFAULT_MAINLINE",
    "PATH_POLICY_DEFAULT_PROFILE",
    "SEQ100_MAINLINE_ID",
    "SEQ100_MAINLINE_PROFILE",
    "ALPHA_PATH20_POLICY_PROFILE",
    "ALPHA_PATH20_POLICY_VERSION",
    "ALPHA_PATH20_SEQUENCE_POLICY_PROFILE",
    "ALPHA_PATH20_SEQUENCE_POLICY_VERSION",
]
