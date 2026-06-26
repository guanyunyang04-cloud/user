from __future__ import annotations

from pathlib import Path
from pkgutil import extend_path


__path__ = extend_path(__path__, __name__)  # type: ignore[name-defined]

_SRC_PACKAGE = Path(__file__).resolve().parent / "src" / "quant_data_platform"
if _SRC_PACKAGE.exists():
    _src_text = str(_SRC_PACKAGE)
    if _src_text not in __path__:
        __path__.append(_src_text)

__version__ = "0.1.0"
