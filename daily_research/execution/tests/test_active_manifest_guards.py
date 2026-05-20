from __future__ import annotations

import pytest


def test_default_dry_run_does_not_require_active_manifest_confirmation() -> None:
    from daily_research.execution.active_manifest_guard import require_active_manifest_write_confirmation

    require_active_manifest_write_confirmation(
        write_requested=False,
        confirmed=False,
        action_flag="--write-active-manifest",
    )


def test_default_no_activate_does_not_require_active_manifest_confirmation() -> None:
    from daily_research.execution.active_manifest_guard import require_active_manifest_write_confirmation

    require_active_manifest_write_confirmation(
        write_requested=False,
        confirmed=False,
        action_flag="--activate-strategy",
    )


def test_unconfirmed_active_manifest_write_is_blocked() -> None:
    from daily_research.execution.active_manifest_guard import require_active_manifest_write_confirmation

    with pytest.raises(RuntimeError, match="--write-active-manifest would modify the active execution manifest"):
        require_active_manifest_write_confirmation(
            write_requested=True,
            confirmed=False,
            action_flag="--write-active-manifest",
        )


def test_confirmed_active_manifest_write_is_allowed() -> None:
    from daily_research.execution.active_manifest_guard import require_active_manifest_write_confirmation

    require_active_manifest_write_confirmation(
        write_requested=True,
        confirmed=True,
        action_flag="--activate-strategy",
    )
