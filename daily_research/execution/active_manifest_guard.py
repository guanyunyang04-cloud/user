from __future__ import annotations


def require_active_manifest_write_confirmation(
    *,
    write_requested: bool,
    confirmed: bool,
    action_flag: str,
) -> None:
    if not bool(write_requested):
        return
    if bool(confirmed):
        return
    flag = str(action_flag or "--write-active-manifest").strip() or "--write-active-manifest"
    raise RuntimeError(
        f"{flag} would modify the active execution manifest. "
        "Pass --confirm-active-manifest-write to make this write explicit."
    )
