import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { HelpPage } from "./HelpPage";
import type { ExecutionApi } from "../types";

function baseApi(overrides: Partial<ExecutionApi> = {}): ExecutionApi {
  return {
    getDailyRunStatus: vi.fn().mockResolvedValue({
      status: "completed",
      latest_verdict: {
        status: "completed",
        target_trading_date: "2026-05-22"
      }
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
    refreshDataSources: vi.fn().mockResolvedValue({ job_id: "data-job", status: "queued", task_name: "data-platform-refresh" }),
    generateTradePlan: vi.fn().mockResolvedValue({ job_id: "plan-job", status: "queued", task_name: "trade-plan" }),
    applyLatestPaperPlan: vi.fn().mockResolvedValue({ status: "ok", apply_result: { filled_order_count: 1 } }),
    ...overrides
  } as unknown as ExecutionApi;
}

describe("HelpPage", () => {
  it("shows the manual workflow and marks the flow complete when fresh plan and account are settled", async () => {
    render(<HelpPage api={baseApi()} />);

    expect(await screen.findByText("手动每日任务")).toBeInTheDocument();
    expect(screen.getByText("今日盘后流程")).toBeInTheDocument();
    expect(screen.getByText("数据、信号、交易计划和模拟账户均已收口。")).toBeInTheDocument();
    expect(screen.getByText("今日流程已完成")).toBeInTheDocument();
    expect(screen.getByText("安全边界")).toBeInTheDocument();
    expect(screen.getByText("不真实下单")).toBeInTheDocument();
    expect(screen.getByText("不自动重训")).toBeInTheDocument();
    expect(screen.getByText("不 promotion")).toBeInTheDocument();
    expect(screen.queryByText(/Windows Task Scheduler/)).not.toBeInTheDocument();
  });

  it("runs daily manual steps directly from the help page", async () => {
    const user = userEvent.setup();
    const api = baseApi();

    render(<HelpPage api={api} />);

    await user.click(await screen.findByRole("button", { name: "1 刷新数据/信号" }));
    await waitFor(() => expect(api.refreshDataSources).toHaveBeenCalled());
    expect(api.refreshDataSources).toHaveBeenCalledWith(expect.objectContaining({
      as_of_date: "2026-05-22",
      universe: "all_a",
      force_unlock: false
    }));

    await user.click(screen.getByRole("button", { name: "2 生成交易计划" }));
    await waitFor(() => expect(api.generateTradePlan).toHaveBeenCalledWith(expect.objectContaining({
      candidate_profile: "active_execution_strategy",
      force_unlock: false
    })));

    await user.click(screen.getByRole("button", { name: "3 模拟账户过账" }));
    await waitFor(() => expect(api.applyLatestPaperPlan).toHaveBeenCalledWith({}));
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
      getDataSources: vi.fn().mockResolvedValue({
        status: "ok",
        lake_root: "lake",
        catalog_status: "ok",
        datasets: [],
        dataset_sync_status: "synced",
        next_refresh_action: "skip",
        next_signal_action: "skip",
        signal_panel_status: "ok",
        signal_panel_latest_date: "2026-05-25",
        data_platform: {
          runs_root: "runs",
          latest_refresh_run: "",
          provider_plan: "formal_free_v3",
          latest_completed_trading_date: "2026-05-25"
        }
      }),
      getTradePlan: vi.fn().mockResolvedValue({
        status: "ok",
        exists: true,
        summary: { signal_date: "2026-05-25", execution_date: "2026-05-25" },
        actions: [{ stock: "002866.SZ", action: "买入" }],
        holdings: [],
        watchlist: [],
        model_info: { signal_panel_status: "ok" },
        paper_trading: { status: "registered", pending_order_count: 1 },
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

  it("asks to settle pending orders before regenerating a stale trade plan", async () => {
    const api = baseApi({
      getDataSources: vi.fn().mockResolvedValue({
        status: "ok",
        lake_root: "lake",
        catalog_status: "ok",
        datasets: [],
        dataset_sync_status: "synced",
        next_refresh_action: "skip",
        next_signal_action: "skip",
        signal_panel_status: "ok",
        signal_panel_latest_date: "2026-05-25",
        data_platform: {
          runs_root: "runs",
          latest_refresh_run: "",
          provider_plan: "formal_free_v3",
          latest_completed_trading_date: "2026-05-25"
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
        paper_trading: { status: "registered", pending_order_count: 3 },
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
        pending_order_count: 3,
        pending_orders: [
          {
            stock: "002866.SZ",
            status: "pending",
            execution_date: "2026-05-25"
          }
        ]
      })
    });

    render(<HelpPage api={api} />);

    expect(await screen.findByText("模拟账户过账")).toBeInTheDocument();
    expect(screen.getByText("交易计划订单已注册，等待可用价格后可在账户页模拟过账。")).toBeInTheDocument();
  });

  it("treats an old trade plan as stale after data and signals reach a newer completed date", async () => {
    const api = baseApi({
      getDataSources: vi.fn().mockResolvedValue({
        status: "ok",
        lake_root: "lake",
        catalog_status: "ok",
        datasets: [],
        dataset_sync_status: "synced",
        next_refresh_action: "skip",
        next_signal_action: "skip",
        signal_panel_status: "ok",
        signal_panel_latest_date: "2026-05-25",
        data_platform: {
          runs_root: "runs",
          latest_refresh_run: "",
          provider_plan: "formal_free_v3",
          latest_completed_trading_date: "2026-05-25"
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
      })
    });

    render(<HelpPage api={api} />);

    expect(await screen.findByText("生成交易计划")).toBeInTheDocument();
    expect(screen.getByText("当前交易计划信号日 2026-05-22，尚未覆盖最新完成交易日 2026-05-25。")).toBeInTheDocument();
  });

  it("treats next-execution-date pending orders as a complete manual flow", async () => {
    const api = baseApi({
      getDataSources: vi.fn().mockResolvedValue({
        status: "ok",
        lake_root: "lake",
        catalog_status: "ok",
        datasets: [],
        dataset_sync_status: "synced",
        next_refresh_action: "skip",
        next_signal_action: "skip",
        signal_panel_status: "ok",
        signal_panel_latest_date: "2026-05-25",
        data_platform: {
          runs_root: "runs",
          latest_refresh_run: "",
          provider_plan: "formal_free_v3",
          latest_completed_trading_date: "2026-05-25"
        }
      }),
      getTradePlan: vi.fn().mockResolvedValue({
        status: "ok",
        exists: true,
        summary: { signal_date: "2026-05-25", execution_date: "2026-05-26" },
        actions: [{ stock: "600664.SH", action: "买入" }],
        holdings: [],
        watchlist: [],
        model_info: { signal_panel_status: "ok" },
        paper_trading: { status: "registered", pending_order_count: 4 },
        txt_preview: [],
        artifact_paths: {}
      }),
      getPaperAccount: vi.fn().mockResolvedValue({
        status: "ok",
        source: "paper_ledger",
        path: "current_positions.csv",
        exists: true,
        available_cash: 474.55,
        positions: [],
        position_count: 3,
        total_shares: 10000,
        last_modified_at: "",
        pending_order_count: 4,
        pending_orders: [
          {
            stock: "600664.SH",
            status: "pending",
            execution_date: "2026-05-26"
          }
        ]
      })
    });

    render(<HelpPage api={api} />);

    expect(await screen.findByText("今日流程已完成")).toBeInTheDocument();
    expect(screen.getByText("数据、信号、交易计划和模拟账户均已收口。")).toBeInTheDocument();
  });
});
