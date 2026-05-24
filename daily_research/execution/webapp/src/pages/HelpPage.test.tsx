import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { HelpPage } from "./HelpPage";
import type { ExecutionApi } from "../types";

function baseApi(overrides: Partial<ExecutionApi> = {}): ExecutionApi {
  return {
    getStatus: vi.fn().mockResolvedValue({
      lock: {},
      current_job: {},
      recent_jobs: []
    }),
    getDataSources: vi.fn().mockResolvedValue({
      status: "ok",
      lake_root: "H:/quant_project/daily_research/output/research_data_lake",
      catalog_status: "ok",
      datasets: [],
      active_dataset_id: "policy_input_bundle__current",
      dataset_sync_status: "synced",
      next_refresh_action: "skip",
      next_signal_action: "skip",
      signal_panel_status: "ok",
      signal_panel_latest_date: "2026-05-22",
      data_platform: {
        runs_root: "runs",
        latest_refresh_run: "",
        provider_plan: "formal_free_v3",
        latest_completed_trading_date: "2026-05-22"
      }
    }),
    getTradePlan: vi.fn().mockResolvedValue({
      status: "ok",
      exists: true,
      summary: { signal_date: "2026-05-22", execution_date: "2026-05-25" },
      actions: [{ stock: "002866.SZ", action: "买入" }],
      holdings: [],
      watchlist: [],
      model_info: { signal_panel_status: "ok" },
      paper_trading: { status: "registered", pending_order_count: 0 },
      txt_preview: [],
      artifact_paths: {}
    }),
    getPaperAccount: vi.fn().mockResolvedValue({
      status: "ok",
      source: "paper_ledger",
      path: "current_positions.csv",
      exists: true,
      available_cash: 100000,
      positions: [],
      position_count: 0,
      total_shares: 0,
      last_modified_at: "",
      pending_order_count: 0,
      pending_orders: []
    }),
    getJobs: vi.fn().mockResolvedValue([]),
    ...overrides
  } as unknown as ExecutionApi;
}

describe("HelpPage", () => {
  it("shows the post-close runbook and says trade-plan generation is ready when data and signal are fresh", async () => {
    render(<HelpPage api={baseApi()} />);

    expect(await screen.findByText("今日盘后流程")).toBeInTheDocument();
    expect(screen.getByText("数据与信号已最新")).toBeInTheDocument();
    expect(screen.getByText("可生成交易计划")).toBeInTheDocument();
    expect(screen.getByText("安全边界")).toBeInTheDocument();
    expect(screen.getByText("不真实下单")).toBeInTheDocument();
    expect(screen.getByText("不自动重训")).toBeInTheDocument();
    expect(screen.getByText("不 promotion")).toBeInTheDocument();
  });

  it("points users to refresh signal panels when dataset is current but signal panels are stale", async () => {
    const api = baseApi({
      getDataSources: vi.fn().mockResolvedValue({
        status: "ok",
        lake_root: "lake",
        catalog_status: "ok",
        datasets: [],
        dataset_sync_status: "synced",
        next_refresh_action: "skip",
        next_signal_action: "refresh",
        signal_panel_status: "stale",
        signal_panel_latest_date: "2026-05-19",
        data_platform: {
          runs_root: "runs",
          latest_refresh_run: "",
          provider_plan: "formal_free_v3",
          latest_completed_trading_date: "2026-05-22"
        }
      })
    });

    render(<HelpPage api={api} />);

    expect(await screen.findByText("先刷新信号面板")).toBeInTheDocument();
    expect(screen.getByText("数据已最新，但 production signal panel 还没覆盖最新完成交易日。")).toBeInTheDocument();
  });

  it("explains pending paper orders caused by missing execution open prices", async () => {
    const api = baseApi({
      getPaperAccount: vi.fn().mockResolvedValue({
        status: "ok",
        source: "paper_ledger",
        path: "current_positions.csv",
        exists: true,
        available_cash: 100000,
        positions: [],
        position_count: 0,
        total_shares: 0,
        last_modified_at: "",
        pending_order_count: 1,
        pending_orders: [
          {
            stock: "002866.SZ",
            status: "pending",
            reason: "missing_execution_open",
            execution_date: "2026-05-25"
          }
        ]
      })
    });

    render(<HelpPage api={api} />);

    expect(await screen.findByText("等待执行日开盘价入湖")).toBeInTheDocument();
    expect(screen.getByText("模拟订单已注册，但执行日 open 价格缺失，所以不能伪造成交。")).toBeInTheDocument();
  });
});
