from __future__ import annotations

import argparse
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PYTHON = Path("C:/Users/ASUS/miniconda3/envs/yolos/python.exe")
DEFAULT_IMPORT_CASES = {
    "numpy_torch_sklearn": ("numpy", "torch", "sklearn"),
    "torch_numpy_sklearn": ("torch", "numpy", "sklearn"),
    "cvxpy_stack": ("cvxpy", "cvxpylayers", "diffcp", "scs"),
    "continuous_policy_model_seq_v3": ("daily_research.continuous_policy.model_seq_v3",),
}
OPENMP_PATTERNS = ("libiomp5md.dll", "vcomp140.dll", "libgomp*.dll")
BACKUP_DIR_MARKERS = (".openmp_conflict_backup",)


@dataclass(frozen=True)
class ImportCase:
    name: str
    modules: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str

    @property
    def ok(self) -> bool:
        return self.returncode == 0

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "modules": list(self.modules),
            "returncode": self.returncode,
            "ok": self.ok,
            "stdout": self.stdout.strip(),
            "stderr": self.stderr.strip(),
        }


def _find_dlls(prefix: Path) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for pattern in OPENMP_PATTERNS:
        hits: list[dict[str, Any]] = []
        for path in sorted(prefix.rglob(pattern)):
            if any(marker in part for part in path.parts for marker in BACKUP_DIR_MARKERS):
                continue
            try:
                stat = path.stat()
            except OSError:
                continue
            hits.append(
                {
                    "path": str(path),
                    "size": stat.st_size,
                    "mtime": int(stat.st_mtime),
                }
            )
        result[pattern] = hits
    return result


def _find_backup_dlls(prefix: Path) -> list[str]:
    backups: list[str] = []
    for pattern in OPENMP_PATTERNS:
        for path in sorted(prefix.rglob(pattern)):
            if any(marker in part for part in path.parts for marker in BACKUP_DIR_MARKERS):
                backups.append(str(path))
    return backups


def _run_import_case(
    name: str,
    modules: tuple[str, ...],
    python_executable: Path,
    timeout: int,
) -> ImportCase:
    imports = "\n".join(f"import {module}" for module in modules)
    code = (
        "import os\n"
        "os.environ.pop('KMP_DUPLICATE_LIB_OK', None)\n"
        f"{imports}\n"
        "print('ok')\n"
    )
    env = os.environ.copy()
    env.pop("KMP_DUPLICATE_LIB_OK", None)
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    proc = subprocess.run(
        [str(python_executable), "-c", code],
        cwd=str(WORKSPACE_ROOT),
        env=env,
        text=True,
        capture_output=True,
        timeout=timeout,
    )
    return ImportCase(
        name=name,
        modules=modules,
        returncode=proc.returncode,
        stdout=proc.stdout,
        stderr=proc.stderr,
    )


def _python_prefix(python_executable: Path, timeout: int) -> Path:
    proc = subprocess.run(
        [
            str(python_executable),
            "-c",
            "import sys; print(sys.prefix)",
        ],
        cwd=str(WORKSPACE_ROOT),
        text=True,
        capture_output=True,
        timeout=timeout,
        check=True,
    )
    return Path(proc.stdout.strip()).resolve()


def _build_report(python_executable: Path, timeout: int) -> dict[str, Any]:
    prefix = _python_prefix(python_executable, timeout=timeout)
    dlls = _find_dlls(prefix)
    backup_dlls = _find_backup_dlls(prefix)
    import_cases = [
        _run_import_case(name, modules, python_executable=python_executable, timeout=timeout)
        for name, modules in DEFAULT_IMPORT_CASES.items()
    ]
    duplicate_intel_openmp = len(dlls.get("libiomp5md.dll", [])) > 1
    mixed_openmp_runtime_families = any(
        dlls.get(pattern) for pattern in ("vcomp140.dll", "libgomp*.dll")
    ) and bool(dlls.get("libiomp5md.dll"))
    import_failures = [case for case in import_cases if not case.ok]
    workaround_set = os.environ.get("KMP_DUPLICATE_LIB_OK")
    return {
        "python_executable": str(python_executable),
        "sys_prefix": str(prefix),
        "conda_prefix": os.environ.get("CONDA_PREFIX"),
        "kmp_duplicate_lib_ok": workaround_set,
        "openmp_runtime_dlls": dlls,
        "openmp_runtime_backup_dlls": backup_dlls,
        "duplicate_intel_openmp_runtime": duplicate_intel_openmp,
        "mixed_openmp_runtime_families": mixed_openmp_runtime_families,
        "import_cases": [case.to_json() for case in import_cases],
        "strict_ok": (
            not duplicate_intel_openmp
            and not import_failures
            and not workaround_set
        ),
        "runtime_family_warning": mixed_openmp_runtime_families,
        "root_fix_required": duplicate_intel_openmp or bool(import_failures),
    }


def _print_text(report: dict[str, Any]) -> None:
    print(f"python: {report['python_executable']}")
    print(f"sys_prefix: {report['sys_prefix']}")
    print(f"KMP_DUPLICATE_LIB_OK: {report['kmp_duplicate_lib_ok']}")
    print(f"duplicate_intel_openmp_runtime: {report['duplicate_intel_openmp_runtime']}")
    print(f"mixed_openmp_runtime_families: {report['mixed_openmp_runtime_families']}")
    print(f"runtime_family_warning: {report['runtime_family_warning']}")
    for pattern, hits in report["openmp_runtime_dlls"].items():
        print(f"{pattern}: {len(hits)}")
        for hit in hits:
            print(f"  - {hit['path']} ({hit['size']} bytes)")
    if report["openmp_runtime_backup_dlls"]:
        print("ignored backup DLLs:")
        for path in report["openmp_runtime_backup_dlls"]:
            print(f"  - {path}")
    print("import cases:")
    for case in report["import_cases"]:
        status = "ok" if case["ok"] else f"failed:{case['returncode']}"
        print(f"  - {case['name']}: {status}")
        if case["stderr"]:
            first_line = case["stderr"].splitlines()[0]
            print(f"    stderr: {first_line}")
    print(f"strict_ok: {report['strict_ok']}")
    print(f"root_fix_required: {report['root_fix_required']}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check whether the active Python environment has OpenMP runtime conflicts."
    )
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    parser.add_argument(
        "--python",
        type=Path,
        default=DEFAULT_PYTHON,
        help="Python executable to inspect; defaults to the project yolos runtime.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero unless no duplicate runtime, no workaround, and import checks pass.",
    )
    parser.add_argument("--timeout", type=int, default=60, help="Per import-case timeout in seconds.")
    args = parser.parse_args()

    python_executable = args.python.resolve()
    report = _build_report(python_executable=python_executable, timeout=args.timeout)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        _print_text(report)
    if args.strict and not report["strict_ok"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
