import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { SystemPage } from "./SystemPage";
import type { ExecutionApi } from "../types";

describe("SystemPage", () => {
  it("shows post-close scheduler status and latest automatic refresh evidence", async () => {
    const api = {
      getStatus: vi.fn().mockResolvedValue({
        runtime_root: "H:/quant_project/daily_research/output/execution_app",
        active_manifest: {
          lake_dataset_id: "policy_input_bundle__current",
          lake_dataset_end_date: "2026-05-22"
        },
        latest_trade_plan: {
          exists: true,
          status: "ok",
          summary: {},
          actions: [],
          holdings: [],
          watchlist: [],
          model_info: {},
          txt_preview: [],
          artifact_paths: {}
        },
        scheduler_status: {
          enabled: true,
          post_close_time: "15:30",
          timezone: "Asia/Shanghai",
          next_check_at: "2026-05-25T15:30:00+08:00",
          missed_status: "none"
        },
        last_auto_refresh: {
          job_id: "auto-refresh-1",
          scheduler_decision: "refresh"
        },
        lock: {}
      }),
      getDoctor: vi.fn().mockResolvedValue({ status: "ok", checked_at: "2026-05-24T00:00:00", checks: [] }),
      unlockRuntime: vi.fn()
    } as unknown as ExecutionApi;

    render(<SystemPage api={api} />);

    expect(await screen.findByText("自动盘后更新")).toBeInTheDocument();
    expect(screen.getAllByText("15:30").length).toBeGreaterThan(0);
    expect(screen.getByText("Asia/Shanghai")).toBeInTheDocument();
    expect(screen.getByText("auto-refresh-1")).toBeInTheDocument();
  });
});
