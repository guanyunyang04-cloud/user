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
  "valuation",
  "industry_concept",
  "money_flow_hotspot"
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
        formal_provider_plan: "formal_free_v3",
        domain_matrix: [
          { provider: "baostock", domain: "market_daily", requirement: "required", supported: true },
          { provider: "tencent_finance", domain: "valuation", requirement: "optional", supported: true }
        ],
        provider_health: {
          status: "unknown",
          summary: {}
        },
        scheduler_status: {
          enabled: true,
          post_close_time: "15:30",
          timezone: "Asia/Shanghai",
          missed_status: ""
        },
        data_platform: {
          runs_root: "runs",
          latest_refresh_run: "",
          provider_plan: "formal_free_v3",
          latest_completed_trading_date: "2026-05-22",
          recommended_domains: recommendedDomains,
          default_refresh: {
            as_of_date: "2026-05-22",
            universe: "all_a",
            domains: recommendedDomains,
            required_domains: recommendedDomains.slice(0, 5),
            provider_plan: "formal_free_v3"
          }
        }
      }),
      runProviderHealth: vi.fn().mockResolvedValue({
        status: "ok",
        provider_plan: "formal_free_v3",
        summary: { ok_domain_count: 2, checked_domain_count: 2 },
        domain_matrix: [],
        providers: []
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
    expect(screen.getByText("Provider Matrix")).toBeInTheDocument();
    expect(screen.getByText("tencent_finance")).toBeInTheDocument();
    expect(screen.getByText("自动更新")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "检查数据源" }));
    await waitFor(() => {
      expect(api.runProviderHealth).toHaveBeenCalledWith({
        as_of_date: "2026-05-22",
        domains: recommendedDomains,
        provider_plan: "formal_free_v3"
      });
    });
    expect(await screen.findByText("provider health: ok")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "补齐到最新交易日" }));

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

  it("shows the current dataset as latest and does not submit a duplicate refresh", async () => {
    const user = userEvent.setup();
    const api = {
      getDataSources: vi.fn().mockResolvedValue({
        status: "ok",
        lake_root: "H:/quant_project/daily_research/output/research_data_lake",
        catalog_status: "ok",
        datasets: [
          {
            dataset_id: "policy_input_bundle__current",
            dataset_kind: "policy_input_bundle",
            domain: "",
            zone: "",
            status: "stored",
            start_date: "2024-01-01",
            end_date: "2026-05-22",
            created_at: "2026-05-24T00:00:00"
          }
        ],
        active_dataset_id: "policy_input_bundle__current",
        latest_policy_input_dataset_id: "policy_input_bundle__current",
        latest_policy_input_dataset_end_date: "2026-05-22",
        dataset_sync_status: "synced",
        current_dataset_status: "latest_complete",
        is_current_dataset_latest: true,
        is_current_dataset_complete: true,
        next_refresh_action: "skip",
        refresh_explanation: "当前 active dataset 已覆盖最新完成交易日 2026-05-22。",
        formal_provider_plan: "formal_free_v3",
        domain_matrix: [],
        provider_health: {},
        scheduler_status: { enabled: true, post_close_time: "15:30", timezone: "Asia/Shanghai" },
        data_platform: {
          runs_root: "runs",
          latest_refresh_run: "",
          provider_plan: "formal_free_v3",
          latest_completed_trading_date: "2026-05-22",
          recommended_domains: recommendedDomains,
          default_refresh: {
            as_of_date: "2026-05-22",
            universe: "all_a",
            domains: recommendedDomains,
            required_domains: recommendedDomains.slice(0, 5),
            provider_plan: "formal_free_v3"
          }
        }
      }),
      runProviderHealth: vi.fn(),
      refreshDataSources: vi.fn()
    } as unknown as ExecutionApi;

    render(<DataPage api={api} />);

    expect(await screen.findByText("当前数据集")).toBeInTheDocument();
    expect(screen.getAllByText("policy_input_bundle__current").length).toBeGreaterThan(0);
    expect(screen.getByText("当前 active dataset 已覆盖最新完成交易日 2026-05-22。")).toBeInTheDocument();
    const button = screen.getByRole("button", { name: "已是最新" });
    expect(button).toBeDisabled();

    await user.click(button);

    expect(api.refreshDataSources).not.toHaveBeenCalled();
    expect(screen.getByText("历史 Lake Datasets")).toBeInTheDocument();
  });

  it("allows a signal-panel refresh when the dataset is current but production signals are stale", async () => {
    const user = userEvent.setup();
    const api = {
      getDataSources: vi.fn().mockResolvedValue({
        status: "ok",
        lake_root: "H:/quant_project/daily_research/output/research_data_lake",
        catalog_status: "ok",
        datasets: [],
        active_dataset_id: "policy_input_bundle__current",
        active_dataset_end_date: "2026-05-22",
        latest_policy_input_dataset_id: "policy_input_bundle__current",
        latest_policy_input_dataset_end_date: "2026-05-22",
        dataset_sync_status: "synced",
        current_dataset_status: "latest_complete",
        is_current_dataset_latest: true,
        is_current_dataset_complete: true,
        next_refresh_action: "skip",
        next_signal_action: "refresh",
        signal_panel_status: "stale",
        signal_panel_latest_date: "2026-05-19",
        signal_panel_target_latest_date: "2026-05-19",
        signal_panel_score_latest_date: "2026-05-19",
        production_anchor_status: "ok",
        refresh_explanation: "数据已最新，但 production signal panels 只到 2026-05-19。",
        signal_panels: {
          required_date: "2026-05-22",
          target_panel_latest_date: "2026-05-19",
          score_panel_latest_date: "2026-05-19",
          target_position_count: 12
        },
        production_anchor: {
          status: "ok",
          fullfit_run_dir: "H:/quant_project/daily_research/output/short_expert_policy_v5b_execalign_production_fullfit_20260421_r1"
        },
        formal_provider_plan: "formal_free_v3",
        domain_matrix: [],
        provider_health: {},
        scheduler_status: { enabled: true, post_close_time: "15:30", timezone: "Asia/Shanghai" },
        data_platform: {
          runs_root: "runs",
          latest_refresh_run: "",
          provider_plan: "formal_free_v3",
          latest_completed_trading_date: "2026-05-22",
          recommended_domains: recommendedDomains,
          default_refresh: {
            as_of_date: "2026-05-22",
            universe: "all_a",
            domains: recommendedDomains,
            required_domains: recommendedDomains.slice(0, 5),
            provider_plan: "formal_free_v3"
          }
        }
      }),
      runProviderHealth: vi.fn(),
      refreshDataSources: vi.fn().mockResolvedValue({ job_id: "signal-job", task_name: "refresh-production-live-panels", status: "queued" }),
      getJob: vi.fn().mockResolvedValue({
        job_id: "signal-job",
        task_name: "refresh-production-live-panels",
        status: "succeeded",
        business_status: "ok",
        artifact_status: "ok",
        metadata: { business_status: "ok", artifact_status: "ok" },
        stdout_tail: ["signal refresh complete"],
        stderr_tail: [],
        can_resume: false
      })
    } as unknown as ExecutionApi;

    render(<DataPage api={api} />);

    expect(await screen.findByText("生产信号面板")).toBeInTheDocument();
    expect(screen.getAllByText("2026-05-19").length).toBeGreaterThan(0);
    const button = screen.getByRole("button", { name: "刷新信号面板" });
    expect(button).not.toBeDisabled();

    await user.click(button);

    await waitFor(() => {
      expect(api.refreshDataSources).toHaveBeenCalledWith({
        as_of_date: "2026-05-22",
        start_date: "",
        universe: "all_a",
        domains: recommendedDomains,
        force_unlock: false
      });
    });
    expect(await screen.findByText("signal refresh complete")).toBeInTheDocument();
  });
});
