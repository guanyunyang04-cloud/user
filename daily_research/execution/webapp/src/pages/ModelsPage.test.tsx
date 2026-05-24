import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { ModelsPage } from "./ModelsPage";
import type { ExecutionApi } from "../types";

function fakeApi(overrides: Partial<ExecutionApi> = {}): ExecutionApi {
  return {
    getStatus: vi.fn(),
    getDoctor: vi.fn(),
    getModels: vi.fn().mockResolvedValue({
      status: "ok",
      updated_at: "2026-05-23 09:00:00",
      models: [
        {
          id: "active-live",
          role: "live",
          name: "active execution",
          status: "active",
          usage: "当前交易计划默认执行链路",
          is_live: true,
          is_shadow: false,
          is_legacy: false,
          data_source: "lake",
          dataset_id: "",
          trained_at: "2026-05-20T18:00:00",
          train_start_date: "",
          train_end_date: "2026-05-20",
          signal_date: "2026-05-22",
          artifact_path: "manifest.json",
          detail: {}
        },
        {
          id: "production-full-fit",
          role: "production",
          name: "production full-fit",
          status: "manual",
          usage: "显式 production retrain 后供交易计划读取",
          is_live: false,
          is_shadow: false,
          is_legacy: false,
          data_source: "lake",
          dataset_id: "",
          trained_at: "",
          train_start_date: "",
          train_end_date: "",
          signal_date: "",
          artifact_path: "",
          detail: {}
        }
      ]
    }),
    getModelDetail: vi.fn().mockResolvedValue({
      status: "ok",
      model: {
        id: "production-full-fit",
        role: "production",
        name: "production full-fit",
        status: "manual",
        usage: "显式 production retrain 后供交易计划读取",
        is_live: false,
        is_shadow: false,
        is_legacy: false,
        data_source: "lake",
        dataset_id: "",
        trained_at: "",
        train_start_date: "",
        train_end_date: "",
        signal_date: "",
        artifact_path: "",
        detail: {}
      }
    }),
    trainModel: vi.fn().mockResolvedValue({ job_id: "job2", status: "queued", task_name: "refresh-production-default" }),
    getDataSources: vi.fn(),
    refreshDataSources: vi.fn(),
    runProviderHealth: vi.fn(),
    getScheduler: vi.fn(),
    updateScheduler: vi.fn(),
    getTradePlan: vi.fn(),
    generateTradePlan: vi.fn(),
    getAccount: vi.fn(),
    saveAccount: vi.fn(),
    resetAccountExample: vi.fn(),
    getPaperAccount: vi.fn(),
    recordPaperCashFlow: vi.fn(),
    recordPaperManualAdjustment: vi.fn(),
    applyLatestPaperPlan: vi.fn(),
    getPaperPerformance: vi.fn(),
    getJobs: vi.fn(),
    getJob: vi.fn(),
    resumeJob: vi.fn(),
    unlockRuntime: vi.fn(),
    ...overrides
  };
}

describe("ModelsPage", () => {
  it("switches to custom dataset and requires confirmation for production training", async () => {
    const user = userEvent.setup();
    const api = fakeApi();
    render(<ModelsPage api={api} />);

    await user.click(await screen.findByRole("button", { name: /production full-fit/i }));
    await user.click(screen.getByLabelText("指定数据集"));
    await user.type(screen.getByLabelText("Dataset ID"), "policy_input_bundle__20260522");
    await user.type(screen.getByLabelText("训练开始"), "20250101");
    await user.type(screen.getByLabelText("训练结束"), "20260522");

    expect(screen.getByRole("button", { name: "启动训练" })).toBeDisabled();

    await user.click(screen.getByLabelText("确认 production 重训"));
    await user.click(screen.getByRole("button", { name: "启动训练" }));

    await waitFor(() => {
      expect(api.trainModel).toHaveBeenCalledWith("production-full-fit", expect.objectContaining({
        dataset_mode: "dataset_id",
        dataset_id: "policy_input_bundle__20260522",
        start_date: "20250101",
        end_date: "20260522"
      }));
    });
  });
});
