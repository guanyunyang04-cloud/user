import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { DataPage } from "./DataPage";
import type { ExecutionApi } from "../types";

const recommendedDomains = [
  "market_daily",
  "trading_calendar",
  "universe_snapshot",
  "security_status",
  "limit_status",
  "industry_concept",
  "valuation"
];

describe("DataPage", () => {
  it("prefills the formal refresh contract and submits it", async () => {
    const user = userEvent.setup();
    const api = {
      getDataSources: vi.fn().mockResolvedValue({
        status: "ok",
        lake_root: "H:/quant_project/daily_research/output/research_data_lake",
        catalog_status: "ok",
        datasets: [],
        data_platform: {
          runs_root: "runs",
          latest_refresh_run: "",
          provider_plan: "default_free",
          latest_completed_trading_date: "2026-05-22",
          recommended_domains: recommendedDomains,
          default_refresh: {
            as_of_date: "2026-05-22",
            universe: "all_a",
            domains: recommendedDomains,
            provider_plan: "default_free"
          }
        }
      }),
      refreshDataSources: vi.fn().mockResolvedValue({ job_id: "job-data", status: "queued" }),
      getJob: vi.fn().mockResolvedValue({
        job_id: "job-data",
        task_name: "data-platform-refresh",
        status: "running",
        metadata: {},
        stdout_tail: ["DataRefresh 4/8 Fetch provider domains"],
        stderr_tail: [],
        can_resume: false
      })
    } as unknown as ExecutionApi;

    render(<DataPage api={api} />);

    expect(await screen.findByLabelText("As-of 日期")).toHaveValue("2026-05-22");
    expect(screen.getByLabelText("Domains")).toHaveValue(recommendedDomains.join(","));

    await user.click(screen.getByRole("button", { name: "刷新数据" }));

    await waitFor(() => {
      expect(api.refreshDataSources).toHaveBeenCalledWith({
        as_of_date: "2026-05-22",
        start_date: "",
        universe: "all_a",
        domains: recommendedDomains,
        force_unlock: false
      });
    });
    expect(await screen.findByText("DataRefresh 4/8 Fetch provider domains")).toBeInTheDocument();
  });
});
