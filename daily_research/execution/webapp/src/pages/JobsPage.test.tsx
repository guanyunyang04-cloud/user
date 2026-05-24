import { act, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
import { JobsPage } from "./JobsPage";
import type { ExecutionApi } from "../types";

describe("JobsPage", () => {
  afterEach(() => {
    vi.useRealTimers();
  });

  it("polls job history and renders the latest status", async () => {
    vi.useFakeTimers();
    const getJobs = vi
      .fn()
      .mockResolvedValueOnce([{ job_id: "job1", task_name: "trade-plan", status: "queued" }])
      .mockResolvedValue([{ job_id: "job1", task_name: "trade-plan", status: "succeeded" }]);
    const api = { getJobs } as unknown as ExecutionApi;

    const { unmount } = render(<JobsPage api={api} pollMs={50} />);

    await act(async () => {
      await Promise.resolve();
    });
    expect(screen.getAllByText("queued").length).toBeGreaterThan(0);

    await act(async () => {
      await vi.advanceTimersByTimeAsync(60);
    });
    expect(screen.getAllByText("succeeded").length).toBeGreaterThan(0);

    expect(getJobs).toHaveBeenCalledTimes(2);
    unmount();
  });

  it("renders a selected job detail with stdout and stderr tails", async () => {
    const api = {
      getJobs: vi.fn().mockResolvedValue([{ job_id: "job1", task_name: "trade-plan", status: "failed" }]),
      getJob: vi.fn().mockResolvedValue({
        job_id: "job1",
        task_name: "trade-plan",
        status: "failed",
        metadata: { job_id: "job1", task_name: "trade-plan", status: "failed", command_argv: ["python", "run.py"] },
        stdout_tail: ["stdout line"],
        stderr_tail: ["stderr line"],
        can_resume: true
      }),
      resumeJob: vi.fn()
    } as unknown as ExecutionApi;

    render(<JobsPage api={api} selectedJobId="job1" pollMs={100000} />);

    expect(await screen.findByText("作业详情")).toBeInTheDocument();
    expect(await screen.findByText("stdout line")).toBeInTheDocument();
    expect(await screen.findByText("stderr line")).toBeInTheDocument();
    expect(api.getJob).toHaveBeenCalledWith("job1", 200);
  });

  it("opens job detail from the jobs list and updates the URL", async () => {
    const user = userEvent.setup();
    window.history.pushState(null, "", "/jobs");
    const api = {
      getJobs: vi.fn().mockResolvedValue([{ job_id: "job2", task_name: "data-platform-refresh", status: "succeeded" }]),
      getJob: vi.fn().mockResolvedValue({
        job_id: "job2",
        task_name: "data-platform-refresh",
        status: "succeeded",
        metadata: {
          job_id: "job2",
          task_name: "data-platform-refresh",
          status: "succeeded",
          command_argv: ["python", "refresh.py"]
        },
        stdout_tail: ["refresh stdout"],
        stderr_tail: ["refresh stderr"],
        can_resume: true
      }),
      resumeJob: vi.fn()
    } as unknown as ExecutionApi;

    render(<JobsPage api={api} pollMs={100000} />);

    await user.click(await screen.findByRole("button", { name: "查看" }));

    expect(await screen.findByText("作业详情")).toBeInTheDocument();
    expect(screen.getByText("refresh stdout")).toBeInTheDocument();
    expect(screen.getByText("refresh stderr")).toBeInTheDocument();
    expect(api.getJob).toHaveBeenCalledWith("job2", 200);
    expect(window.location.pathname).toBe("/jobs/job2");
    expect(screen.getAllByRole("button", { name: "重跑/恢复 job2" }).length).toBeGreaterThan(0);
  });

  it("shows business and runner status separately when a job was reconciled", async () => {
    const api = {
      getJobs: vi.fn().mockResolvedValue([
        {
          job_id: "job1",
          task_name: "data-platform-refresh",
          status: "succeeded",
          business_status: "ok",
          runner_status: "warning"
        }
      ]),
      getJob: vi.fn().mockResolvedValue({
        job_id: "job1",
        task_name: "data-platform-refresh",
        status: "succeeded",
        business_status: "ok",
        runner_status: "warning",
        artifact_status: "ok",
        evidence_paths: { refresh_manifest: "H:/run/refresh_manifest.json" },
        metadata: {
          job_id: "job1",
          task_name: "data-platform-refresh",
          status: "succeeded",
          business_status: "ok",
          runner_status: "warning",
          artifact_status: "ok",
          evidence_paths: { refresh_manifest: "H:/run/refresh_manifest.json" },
          command_argv: ["python", "refresh.py"]
        },
        stdout_tail: ["ok manifest"],
        stderr_tail: ["runner warning"],
        can_resume: false
      }),
      resumeJob: vi.fn()
    } as unknown as ExecutionApi;

    render(<JobsPage api={api} selectedJobId="job1" pollMs={100000} />);

    expect(await screen.findByText("业务状态")).toBeInTheDocument();
    expect(screen.getAllByText("ok").length).toBeGreaterThan(0);
    expect(screen.getByText("Runner")).toBeInTheDocument();
    expect(screen.getAllByText("warning").length).toBeGreaterThan(0);
    expect(screen.getByText("H:/run/refresh_manifest.json")).toBeInTheDocument();
  });
});
