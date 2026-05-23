import { act, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { TradePlanPage } from "./TradePlanPage";
import type { ExecutionApi } from "../types";

describe("TradePlanPage", () => {
  afterEach(() => {
    vi.useRealTimers();
  });

  it("shows a missing state when trade plan artifacts do not exist", async () => {
    const api = {
      getTradePlan: vi.fn().mockResolvedValue({
        status: "missing",
        exists: false,
        summary: {},
        actions: [],
        holdings: [],
        watchlist: [],
        model_info: {},
        txt_preview: [],
        artifact_paths: {}
      }),
      generateTradePlan: vi.fn()
    } as unknown as ExecutionApi;

    render(<TradePlanPage api={api} />);

    expect(await screen.findByText("交易计划未生成")).toBeInTheDocument();
  });

  it("polls a generated trade-plan job and refreshes artifacts after completion", async () => {
    vi.useFakeTimers();
    const getTradePlan = vi
      .fn()
      .mockResolvedValueOnce({
        status: "missing",
        exists: false,
        summary: {},
        actions: [],
        holdings: [],
        watchlist: [],
        model_info: {},
        txt_preview: [],
        artifact_paths: {}
      })
      .mockResolvedValue({
        status: "ok",
        exists: true,
        summary: { signal_date: "2026-05-22" },
        actions: [{ stock: "600000.SH", action: "buy", target_weight: "0.1" }],
        holdings: [],
        watchlist: [],
        model_info: { trading_day_lag: 20 },
        txt_preview: ["plan ready"],
        artifact_paths: {}
      });
    const getJob = vi
      .fn()
      .mockResolvedValueOnce({
        job_id: "job-trade",
        task_name: "trade-plan",
        status: "running",
        metadata: {},
        stdout_tail: ["building plan"],
        stderr_tail: [],
        can_resume: false
      })
      .mockResolvedValue({
        job_id: "job-trade",
        task_name: "trade-plan",
        status: "succeeded",
        metadata: { business_status: "ok", runner_status: "ok" },
        stdout_tail: ["plan ready"],
        stderr_tail: [],
        can_resume: false
      });
    const api = {
      getTradePlan,
      generateTradePlan: vi.fn().mockResolvedValue({ job_id: "job-trade", status: "queued" }),
      getJob
    } as unknown as ExecutionApi;

    render(<TradePlanPage api={api} pollMs={50} />);

    await act(async () => {
      await Promise.resolve();
    });
    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: "生成交易计划" }));
      await Promise.resolve();
      await Promise.resolve();
    });
    expect(screen.getByText("building plan")).toBeInTheDocument();

    await act(async () => {
      await vi.advanceTimersByTimeAsync(60);
      await Promise.resolve();
    });

    expect(getTradePlan).toHaveBeenCalledTimes(2);
    expect(screen.getAllByText("plan ready").length).toBeGreaterThan(0);
    expect(screen.getByText("600000.SH")).toBeInTheDocument();
  });
});
