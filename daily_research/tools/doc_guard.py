from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_DOCS = [
    "brain/master_brain.md",
    "brain/brain_architecture.md",
    "brain/working_memory.md",
    "brain/procedural_memory.md",
    "brain/environment_model.md",
    "brain/brain_manifest.json",
    "daily_research/brain/semantic_memory.md",
    "daily_research/brain/brain_architecture.md",
    "daily_research/brain/project_map.md",
    "daily_research/brain/working_memory.md",
    "daily_research/brain/procedural_memory.md",
    "daily_research/brain/environment_model.md",
    "daily_research/brain/action_system.md",
    "daily_research/brain/episodic_memory.md",
    "daily_research/brain/brain_manifest.json",
    "t0_project/brain/semantic_memory.md",
    "t0_project/brain/brain_architecture.md",
    "t0_project/brain/working_memory.md",
    "t0_project/brain/procedural_memory.md",
    "t0_project/brain/environment_model.md",
    "t0_project/brain/action_system.md",
    "t0_project/brain/episodic_memory.md",
    "t0_project/brain/brain_manifest.json",
    "daily_stock_analysis-main/brain/semantic_memory.md",
    "daily_stock_analysis-main/brain/brain_architecture.md",
    "daily_stock_analysis-main/brain/working_memory.md",
    "daily_stock_analysis-main/brain/procedural_memory.md",
    "daily_stock_analysis-main/brain/environment_model.md",
    "daily_stock_analysis-main/brain/action_system.md",
    "daily_stock_analysis-main/brain/episodic_memory.md",
    "daily_stock_analysis-main/brain/brain_manifest.json",
]


@dataclass(frozen=True)
class DocRule:
    max_lines: int | None = None
    forbidden_heading_patterns: tuple[tuple[str, str], ...] = ()
    enforce_non_decreasing_dated_headings: bool = False


DOC_RULES = {
    "brain/master_brain.md": DocRule(
        max_lines=180,
        forbidden_heading_patterns=(
            (r"^##\s+20\d{2}-\d{2}-\d{2}\b", "main brain should stay a current workspace memory instead of becoming a dated log"),
        ),
    ),
    "brain/brain_architecture.md": DocRule(
        max_lines=180,
        forbidden_heading_patterns=(
            (r"^##\s+20\d{2}-\d{2}-\d{2}\b", "main brain architecture should stay structural instead of becoming a dated log"),
        ),
    ),
    "brain/working_memory.md": DocRule(
        max_lines=160,
        forbidden_heading_patterns=(
            (r"^##\s+20\d{2}-\d{2}-\d{2}\b", "main working memory should stay a current-priority document instead of becoming a dated log"),
        ),
    ),
    "brain/procedural_memory.md": DocRule(
        max_lines=180,
        forbidden_heading_patterns=(
            (r"^##\s+20\d{2}-\d{2}-\d{2}\b", "main procedural memory should stay a reusable methods document instead of becoming a dated log"),
        ),
    ),
    "brain/environment_model.md": DocRule(
        max_lines=160,
        forbidden_heading_patterns=(
            (r"^##\s+20\d{2}-\d{2}-\d{2}\b", "main environment model should stay an environment baseline instead of becoming a dated log"),
        ),
    ),
    "brain/brain_manifest.json": DocRule(
        max_lines=200,
    ),
    "daily_research/brain/semantic_memory.md": DocRule(
        max_lines=180,
        forbidden_heading_patterns=(
            (r"^##\s+20\d{2}-\d{2}-\d{2}\b", "daily_research semantic memory should stay a current-state document instead of becoming a dated log"),
        ),
    ),
    "daily_research/brain/brain_architecture.md": DocRule(
        max_lines=180,
        forbidden_heading_patterns=(
            (r"^##\s+20\d{2}-\d{2}-\d{2}\b", "daily_research brain architecture should stay structural instead of becoming a dated log"),
        ),
    ),
    "daily_research/brain/project_map.md": DocRule(
        max_lines=280,
        forbidden_heading_patterns=(
            (r"^##\s+20\d{2}-\d{2}-\d{2}\b", "daily_research project_map should stay a collaboration guide instead of becoming a dated log"),
        ),
    ),
    "daily_research/brain/working_memory.md": DocRule(
        max_lines=260,
        forbidden_heading_patterns=(
            (r"^##\s+20\d{2}-\d{2}-\d{2}\b", "daily_research working memory should stay a current-decision document instead of becoming a dated log"),
        ),
    ),
    "daily_research/brain/procedural_memory.md": DocRule(
        max_lines=220,
        forbidden_heading_patterns=(
            (r"^##\s+20\d{2}-\d{2}-\d{2}\b", "daily_research procedural memory should stay a reusable methods document instead of becoming a dated log"),
        ),
    ),
    "daily_research/brain/environment_model.md": DocRule(
        max_lines=180,
        forbidden_heading_patterns=(
            (r"^##\s+20\d{2}-\d{2}-\d{2}\b", "daily_research environment model should stay an environment baseline instead of becoming a dated log"),
        ),
    ),
    "daily_research/brain/action_system.md": DocRule(
        max_lines=220,
        forbidden_heading_patterns=(
            (r"^##\s+20\d{2}-\d{2}-\d{2}\b", "daily_research action system should stay an operations guide instead of becoming a dated log"),
        ),
    ),
    "daily_research/brain/episodic_memory.md": DocRule(
        enforce_non_decreasing_dated_headings=True,
    ),
    "daily_research/brain/brain_manifest.json": DocRule(
        max_lines=220,
    ),
    "t0_project/brain/semantic_memory.md": DocRule(
        max_lines=160,
        forbidden_heading_patterns=(
            (r"^##\s+20\d{2}-\d{2}-\d{2}\b", "t0 semantic memory should stay a current-state document instead of becoming a dated log"),
        ),
    ),
    "t0_project/brain/brain_architecture.md": DocRule(
        max_lines=160,
        forbidden_heading_patterns=(
            (r"^##\s+20\d{2}-\d{2}-\d{2}\b", "t0 brain architecture should stay structural instead of becoming a dated log"),
        ),
    ),
    "t0_project/brain/working_memory.md": DocRule(
        max_lines=160,
        forbidden_heading_patterns=(
            (r"^##\s+20\d{2}-\d{2}-\d{2}\b", "t0 working memory should stay a current-priority document instead of becoming a dated log"),
        ),
    ),
    "t0_project/brain/procedural_memory.md": DocRule(
        max_lines=160,
        forbidden_heading_patterns=(
            (r"^##\s+20\d{2}-\d{2}-\d{2}\b", "t0 procedural memory should stay a reusable methods document instead of becoming a dated log"),
        ),
    ),
    "t0_project/brain/environment_model.md": DocRule(
        max_lines=160,
        forbidden_heading_patterns=(
            (r"^##\s+20\d{2}-\d{2}-\d{2}\b", "t0 environment model should stay an environment baseline instead of becoming a dated log"),
        ),
    ),
    "t0_project/brain/action_system.md": DocRule(
        max_lines=180,
        forbidden_heading_patterns=(
            (r"^##\s+20\d{2}-\d{2}-\d{2}\b", "t0 action system should stay an operations guide instead of becoming a dated log"),
        ),
    ),
    "t0_project/brain/episodic_memory.md": DocRule(
        max_lines=120,
        forbidden_heading_patterns=(
            (r"^##\s+20\d{2}-\d{2}-\d{2}\b", "t0 episodic memory should stay empty or time-ordered once activated"),
        ),
    ),
    "t0_project/brain/brain_manifest.json": DocRule(
        max_lines=180,
    ),
    "daily_stock_analysis-main/brain/semantic_memory.md": DocRule(
        max_lines=180,
        forbidden_heading_patterns=(
            (r"^##\s+20\d{2}-\d{2}-\d{2}\b", "daily_stock_analysis semantic memory should stay a current-state document instead of becoming a dated log"),
        ),
    ),
    "daily_stock_analysis-main/brain/brain_architecture.md": DocRule(
        max_lines=180,
        forbidden_heading_patterns=(
            (r"^##\s+20\d{2}-\d{2}-\d{2}\b", "daily_stock_analysis brain architecture should stay structural instead of becoming a dated log"),
        ),
    ),
    "daily_stock_analysis-main/brain/working_memory.md": DocRule(
        max_lines=160,
        forbidden_heading_patterns=(
            (r"^##\s+20\d{2}-\d{2}-\d{2}\b", "daily_stock_analysis working memory should stay a current-priority document instead of becoming a dated log"),
        ),
    ),
    "daily_stock_analysis-main/brain/procedural_memory.md": DocRule(
        max_lines=200,
        forbidden_heading_patterns=(
            (r"^##\s+20\d{2}-\d{2}-\d{2}\b", "daily_stock_analysis procedural memory should stay a reusable methods document instead of becoming a dated log"),
        ),
    ),
    "daily_stock_analysis-main/brain/environment_model.md": DocRule(
        max_lines=180,
        forbidden_heading_patterns=(
            (r"^##\s+20\d{2}-\d{2}-\d{2}\b", "daily_stock_analysis environment model should stay an environment baseline instead of becoming a dated log"),
        ),
    ),
    "daily_stock_analysis-main/brain/action_system.md": DocRule(
        max_lines=220,
        forbidden_heading_patterns=(
            (r"^##\s+20\d{2}-\d{2}-\d{2}\b", "daily_stock_analysis action system should stay an operations guide instead of becoming a dated log"),
        ),
    ),
    "daily_stock_analysis-main/brain/episodic_memory.md": DocRule(
        max_lines=120,
        forbidden_heading_patterns=(
            (r"^##\s+20\d{2}-\d{2}-\d{2}\b", "daily_stock_analysis episodic memory should stay empty or time-ordered once activated"),
        ),
    ),
    "daily_stock_analysis-main/brain/brain_manifest.json": DocRule(
        max_lines=220,
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


def _dated_headings(lines: Iterable[str]) -> list[tuple[int, str, str]]:
    regex = re.compile(r"^##\s+(20\d{2}-\d{2}-\d{2})\b")
    headings: list[tuple[int, str, str]] = []
    for lineno, line in enumerate(lines, start=1):
        match = regex.search(line)
        if match:
            headings.append((lineno, match.group(1), line))
    return headings


def _check_manifest_semantics(path: Path, text: str) -> list[str]:
    issues: list[str] = []
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        return [f"invalid_json={exc}"]

    normalized = _normalized_path(path)
    if normalized == "brain/brain_manifest.json":
        if data.get("brain_type") != "main":
            issues.append("main_manifest_brain_type_must_be_main")
        child_brains = data.get("child_brains")
        if not isinstance(child_brains, list) or not child_brains:
            issues.append("main_manifest_child_brains_missing_or_empty")
            return issues

        for child in child_brains:
            if not isinstance(child, dict):
                issues.append("main_manifest_child_entry_must_be_object")
                continue
            for key in ("id", "path", "body_root", "entrypoint", "attach_status"):
                if key not in child:
                    issues.append(f"main_manifest_child_missing_key:{key}")

            child_path = child.get("path")
            if not isinstance(child_path, str):
                continue
            child_manifest = Path(child_path)
            if not child_manifest.exists():
                issues.append(f"child_manifest_missing:{child_path}")
                continue

            try:
                child_data = json.loads(_read_text(child_manifest))
            except json.JSONDecodeError as exc:
                issues.append(f"child_manifest_invalid_json:{child_path}:{exc}")
                continue

            if child_data.get("brain_id") != child.get("id"):
                issues.append(f"child_brain_id_mismatch:{child_path}")
            if child_data.get("parent_brain") != "brain/brain_manifest.json":
                issues.append(f"child_parent_brain_mismatch:{child_path}")
            if child_data.get("body_root") != child.get("body_root"):
                issues.append(f"child_body_root_mismatch:{child_path}")
            if child_data.get("entrypoint") != child.get("entrypoint"):
                issues.append(f"child_entrypoint_mismatch:{child_path}")
            if not str(child_data.get("attach_status", "")).startswith("attached"):
                issues.append(f"child_attach_status_invalid:{child_path}")
        return issues

    if normalized.endswith("/brain/brain_manifest.json"):
        required = {
            "brain_type",
            "brain_id",
            "parent_brain",
            "attach_status",
            "entrypoint",
            "body_root",
            "read_order",
            "write_routes",
            "body_map",
            "modules",
            "handoff_contract",
        }
        missing = sorted(required.difference(data))
        if missing:
            issues.append(f"sub_manifest_missing_keys:{','.join(missing)}")
            return issues

        if data.get("brain_type") != "sub_brain":
            issues.append("sub_manifest_brain_type_must_be_sub_brain")
        if not str(data.get("attach_status", "")).startswith("attached"):
            issues.append("sub_manifest_attach_status_must_start_with_attached")

        parent_path = Path(str(data.get("parent_brain", "")))
        if not parent_path.exists():
            issues.append(f"parent_brain_missing:{parent_path.as_posix()}")

        body_root = Path(str(data.get("body_root", "")))
        if not body_root.exists():
            issues.append(f"body_root_missing:{body_root.as_posix()}")

        entrypoint = Path(str(data.get("entrypoint", "")))
        if not entrypoint.exists():
            issues.append(f"entrypoint_missing:{entrypoint.as_posix()}")

        read_order = data.get("read_order")
        if not isinstance(read_order, list) or not read_order:
            issues.append("read_order_missing_or_empty")
        else:
            if Path(str(read_order[0])) != entrypoint:
                issues.append("read_order_first_item_should_match_entrypoint")
            for item in read_order:
                if not Path(str(item)).exists():
                    issues.append(f"read_order_path_missing:{item}")

        write_routes = data.get("write_routes")
        if not isinstance(write_routes, dict) or not write_routes:
            issues.append("write_routes_missing_or_empty")
        else:
            for route_path in write_routes.values():
                if not Path(str(route_path)).exists():
                    issues.append(f"write_route_missing:{route_path}")

        body_map = data.get("body_map")
        if not isinstance(body_map, dict) or not body_map:
            issues.append("body_map_missing_or_empty")
        else:
            for mapped_paths in body_map.values():
                if not isinstance(mapped_paths, list) or not mapped_paths:
                    issues.append("body_map_group_missing_paths")
                    continue
                for mapped_path in mapped_paths:
                    if not Path(str(mapped_path)).exists():
                        issues.append(f"body_map_path_missing:{mapped_path}")

        modules = data.get("modules")
        if not isinstance(modules, list) or not modules:
            issues.append("modules_missing_or_empty")
        else:
            for module in modules:
                if not isinstance(module, dict):
                    issues.append("module_entry_must_be_object")
                    continue
                if "id" not in module or "paths" not in module:
                    issues.append("module_missing_id_or_paths")
                    continue
                paths = module.get("paths")
                if not isinstance(paths, list) or not paths:
                    issues.append(f"module_paths_missing_or_empty:{module.get('id', 'unknown')}")
                    continue
                for module_path in paths:
                    if not Path(str(module_path)).exists():
                        issues.append(f"module_path_missing:{module_path}")

        handoff = data.get("handoff_contract")
        if not isinstance(handoff, dict):
            issues.append("handoff_contract_missing_or_invalid")
        else:
            entry_sequence = handoff.get("entry_sequence")
            if not isinstance(entry_sequence, list) or not entry_sequence:
                issues.append("handoff_entry_sequence_missing_or_empty")
            else:
                if Path(str(entry_sequence[0])) != path:
                    issues.append("handoff_entry_sequence_should_start_with_manifest")
                for item in entry_sequence:
                    if not Path(str(item)).exists():
                        issues.append(f"handoff_entry_path_missing:{item}")

    return issues


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

            if rule.enforce_non_decreasing_dated_headings:
                headings = _dated_headings(lines)
                out_of_order_pairs: list[tuple[tuple[int, str, str], tuple[int, str, str]]] = []
                for previous, current in zip(headings, headings[1:]):
                    if current[1] < previous[1]:
                        out_of_order_pairs.append((previous, current))

                print(f"  dated_heading_order_issues={len(out_of_order_pairs)}")
                if out_of_order_pairs:
                    has_issue = True
                    for previous, current in out_of_order_pairs[: args.show_lines]:
                        print(f"    ! {path}:{current[0]} date {current[1]} appears after later date {previous[1]}")
                        print(f"      prev={previous[2]}")
                        print(f"      curr={current[2]}")

        if path.suffix.lower() == ".json" and _normalized_path(path).endswith("brain_manifest.json"):
            manifest_issues = _check_manifest_semantics(path, text)
            print(f"  manifest_semantic_issues={len(manifest_issues)}")
            if manifest_issues:
                has_issue = True
                for issue in manifest_issues[: args.show_lines]:
                    print(f"    ! {issue}")

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
    parser = argparse.ArgumentParser(description="UTF-8-safe and structure-aware helper for workspace brains.")
    sub = parser.add_subparsers(dest="command", required=True)

    check = sub.add_parser("check", help="Check brain docs for encoding damage and structure drift.")
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
