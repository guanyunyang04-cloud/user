"""Read-only adapter for the local TDX tqcenter data gateway."""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any


DEFAULT_TQCENTER_PATH = Path(r"H:\new_tdx64\PYPlugins\user\t0_project\tqcenter.py")


class TqDataSourceError(RuntimeError):
    """Raised when the tqcenter data gateway cannot be used safely."""


@dataclass(frozen=True)
class TqDataSourceConfig:
    tqcenter_path: Path = DEFAULT_TQCENTER_PATH
    initialize_path: Path | None = None


class TqDataSource:
    """Context-managed, read-only wrapper around `tqcenter.py`."""

    def __init__(self, config: TqDataSourceConfig | None = None) -> None:
        self.config = config or TqDataSourceConfig()
        self._module: ModuleType | None = None
        self._tq: Any | None = None

    @property
    def tqcenter_path(self) -> Path:
        return self.config.tqcenter_path

    @property
    def tq(self) -> Any:
        if self._tq is None:
            raise TqDataSourceError("TQ data source is not initialized")
        return self._tq

    @property
    def dll_path(self) -> str:
        if self._module is None:
            return ""
        return str(getattr(self._module, "global_dll_path", ""))

    @property
    def run_id(self) -> int:
        return int(getattr(self.tq, "run_id", -1))

    @property
    def run_mode(self) -> int:
        return int(getattr(self.tq, "run_mode", -1))

    def __enter__(self) -> "TqDataSource":
        self.open()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()

    def open(self) -> None:
        path = self.tqcenter_path
        if not path.exists():
            raise TqDataSourceError(f"tqcenter.py not found: {path}")
        try:
            spec = importlib.util.spec_from_file_location("traditional_quant_tqcenter", path)
            if spec is None or spec.loader is None:
                raise TqDataSourceError(f"cannot load tqcenter spec: {path}")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            tq = getattr(module, "tq")
            init_path = self.config.initialize_path or path
            tq.initialize(str(init_path))
        except Exception as exc:  # noqa: BLE001 - wrap DLL/import errors with context.
            raise TqDataSourceError(f"failed to initialize tqcenter data source: {exc}") from exc
        self._module = module
        self._tq = tq

    def close(self) -> None:
        if self._tq is None:
            return
        try:
            self._tq.close()
        finally:
            self._tq = None
            self._module = None

    def get_stock_list(self) -> list[str]:
        try:
            return list(self.tq.get_stock_list())
        except Exception as exc:  # noqa: BLE001
            raise TqDataSourceError(f"failed to fetch stock list: {exc}") from exc

    def get_stock_info(self, code: str) -> dict[str, Any]:
        try:
            payload = self.tq.get_stock_info(code)
        except Exception as exc:  # noqa: BLE001
            raise TqDataSourceError(f"failed to fetch stock info for {code}: {exc}") from exc
        if not isinstance(payload, dict):
            raise TqDataSourceError(f"stock info for {code} is not a dict")
        return payload

    def get_daily_bars(
        self,
        codes: list[str],
        *,
        start_time: str,
        end_time: str,
    ) -> dict[str, Any]:
        if not codes:
            return {}
        try:
            payload = self.tq.get_market_data(
                field_list=[],
                stock_list=codes,
                period="1d",
                start_time=start_time,
                end_time=end_time,
                count=-1,
                dividend_type="none",
                fill_data=False,
            )
        except Exception as exc:  # noqa: BLE001
            raise TqDataSourceError(f"failed to fetch daily bars for batch: {exc}") from exc
        if not isinstance(payload, dict):
            raise TqDataSourceError("daily bars payload is not a dict")
        return payload

    def get_trading_dates(
        self,
        *,
        market: str = "SH",
        start_time: str,
        end_time: str,
    ) -> list[str]:
        try:
            dates = self.tq.get_trading_dates(market=market, start_time=start_time, end_time=end_time)
        except Exception as exc:  # noqa: BLE001
            raise TqDataSourceError(f"failed to fetch trading dates: {exc}") from exc
        return list(dates)
