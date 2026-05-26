import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { SystemPage } from "./SystemPage";
import type { ExecutionApi } from "../types";

describe("SystemPage", () => {
  it("shows post-close scheduler status and latest automatic refresh evidence", async () => {
    const api = {
      getDailyRunStatus: vi.fn().mockResolvedValue({
        status: "completed",
        daily_runs_root: "H:/quant_project/daily_research/output/execution_app/daily_runs",
        latest_verdict: {
          status: "completed",
          run_date: "2026-05-26",
          target_trading_date: "2026-05-26",
          dataset_id: "policy_input_bundle__current",
          signal_panel_date: "2026-05-26",
          trade_plan_run_dir: "H:/trade_plan_run",
          paper_reconcile_status: "ok",
          evidence_paths: { verdict: "H:/verdict.json" }
        }
      }),
      getSystemDoctor: vi.fn().mockResolvedValue({ status: "ok", checked_at: "2026-05-24T00:00:00", checks: [] }),
      getScheduler: vi.fn().mockResolvedValue({
        status: "ok",
        installed: true,
        enabled: true,
        task_name: "DailyResearchDailyPlan",
        next_run_time: "2026-05-27 15:45:00",
        last_result: "0"
      }),
      unlockRuntime: vi.fn()
    } as unknown as ExecutionApi;

    render(<SystemPage api={api} />);

    expect(await screen.findByText("Windows Task Scheduler")).toBeInTheDocument();
    expect(screen.getAllByText("DailyResearchDailyPlan").length).toBeGreaterThan(0);
    expect(screen.getByText("policy_input_bundle__current")).toBeInTheDocument();
    expect(screen.getByText("H:/verdict.json")).toBeInTheDocument();
  });
});
