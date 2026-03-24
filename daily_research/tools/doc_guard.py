from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_DOCS = [
    "README.md",
    "daily_research/README.md",
    "daily_research/research_log.md",
    "daily_research/daily_research_plan.md",
]


@dataclass(frozen=True)
class DocRule:
    max_lines: int | None = None
    forbidden_heading_patterns: tuple[tuple[str, str], ...] = ()


DOC_RULES = {
    "README.md": DocRule(
        max_lines=200,
        forbidden_heading_patterns=(
            (r"^##\s+20\d{2}-\d{2}-\d{2}\b", "root README should not grow into a dated change log"),
        ),
    ),
    "daily_research/README.md": DocRule(
        max_lines=220,
        forbidden_heading_patterns=(
            (r"^##\s+20\d{2}-\d{2}-\d{2}\b", "daily_research README should only describe the current state and entrypoints"),
        ),
    ),
    "daily_research/daily_research_plan.md": DocRule(
        max_lines=220,
        forbidden_heading_patterns=(
            (r"^##\s+20\d{2}-\d{2}-\d{2}\b", "daily_research_plan should keep only current priorities and stop rules"),
        ),
    ),
}


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig")


def _tail_lines(text: str, count: int) -> list[str]:
    lines = text.splitlines()
    return lines[-count:] if count > 0 else lines


def _suspicious_question_lines(lines: Iterable[str]) -> list[str]:
    out: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("```"):
            continue
        if "?" in line:
            out.append(line)
    return out


def _normalized_path(path: Path) -> str:
    return path.as_posix()


def _resolve_rule(path: Path) -> DocRule | None:
    normalized = _normalized_path(path)
    for suffix, rule in sorted(DOC_RULES.items(), key=lambda item: len(item[0]), reverse=True):
        if normalized.endswith(suffix):
            return rule
    return None


def _matching_lines(lines: Iterable[str], pattern: str) -> list[str]:
    regex = re.compile(pattern)
    return [line for line in lines if regex.search(line)]


def cmd_check(args: argparse.Namespace) -> int:
    has_issue = False
    for raw in args.files:
        path = Path(raw)
        if not path.exists():
            print(f"[missing] {path}")
            has_issue = True
            continue

        text = _read_text(path)
        lines = text.splitlines()
        line_count = len(lines)
        replacement_count = text.count("\ufffd")
        tail = _tail_lines(text, args.tail_lines)
        tail_question_lines = _suspicious_question_lines(tail)
        rule = _resolve_rule(path)

        print(f"[check] {path}")
        print(f"  line_count={line_count}")
        print(f"  replacement_char_count={replacement_count}")
        print(f"  suspicious_question_lines_in_tail={len(tail_question_lines)}")

        if replacement_count or tail_question_lines:
            has_issue = True
            for line in tail_question_lines[: args.show_lines]:
                print(f"    ? {line}")

        if rule and rule.max_lines is not None and line_count > rule.max_lines:
            has_issue = True
            print(f"  structural_issue=line_count_exceeds_limit ({line_count} > {rule.max_lines})")

        if rule:
            for pattern, reason in rule.forbidden_heading_patterns:
                matches = _matching_lines(lines, pattern)
                print(f"  forbidden_heading_matches={len(matches)} for rule: {reason}")
                if matches:
                    has_issue = True
                    for line in matches[: args.show_lines]:
                        print(f"    ! {line}")

    return 1 if has_issue else 0


def cmd_append(args: argparse.Namespace) -> int:
    path = Path(args.file)
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = _read_text(path) if path.exists() else ""
    body = Path(args.body_file).read_text(encoding="utf-8")
    parts = [existing.rstrip(), args.header.rstrip(), "", body.strip(), ""]
    path.write_text("\n".join(part for part in parts if part != ""), encoding="utf-8")
    print(f"[append] wrote UTF-8 section to {path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="UTF-8-safe and structure-aware helper for workspace docs.")
    sub = parser.add_subparsers(dest="command", required=True)

    check = sub.add_parser("check", help="Check docs for encoding damage and structure drift.")
    check.add_argument("--files", nargs="+", default=DEFAULT_DOCS)
    check.add_argument("--tail-lines", type=int, default=120)
    check.add_argument("--show-lines", type=int, default=12)
    check.set_defaults(func=cmd_check)

    append = sub.add_parser("append", help="Append a markdown section using explicit UTF-8 writes.")
    append.add_argument("--file", required=True)
    append.add_argument("--header", required=True, help="Markdown header line, e.g. ## 2026-03-20 ...")
    append.add_argument("--body-file", required=True, help="UTF-8 markdown fragment to append under the header.")
    append.set_defaults(func=cmd_append)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
