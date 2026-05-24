import { render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { App } from "./App";
import type { ExecutionApi } from "./types";

function fakeApi(): ExecutionApi {
  return {
    getStatus: vi.fn().mockResolvedValue({ lock: {}, current_job: {}, recent_jobs: [] }),
    getDataSources: vi.fn().mockResolvedValue({
      status: "ok",
      lake_root: "lake",
      catalog_status: "ok",
      datasets: [],
      dataset_sync_status: "synced",
      next_refresh_action: "skip",
      next_signal_action: "skip",
      signal_panel_status: "ok",
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
      summary: {},
      actions: [],
      holdings: [],
      watchlist: [],
      model_info: {},
      txt_preview: [],
      artifact_paths: {}
    }),
    getPaperAccount: vi.fn().mockResolvedValue({
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
    getJobs: vi.fn().mockResolvedValue([])
  } as unknown as ExecutionApi;
}

describe("App help routing", () => {
  afterEach(() => {
    window.history.pushState(null, "", "/");
  });

  it("shows HelpPage from /help and adds 帮助 to navigation without restoring 教程", async () => {
    window.history.pushState(null, "", "/help");

    render(<App api={fakeApi()} />);

    expect(await screen.findByRole("heading", { name: "帮助" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "帮助" })).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "教程" })).not.toBeInTheDocument();
  });

  it("maps the legacy /guide URL to the same HelpPage", async () => {
    window.history.pushState(null, "", "/guide");

    render(<App api={fakeApi()} />);

    expect(await screen.findByRole("heading", { name: "帮助" })).toBeInTheDocument();
    expect(screen.getByText("今日盘后流程")).toBeInTheDocument();
  });
});
