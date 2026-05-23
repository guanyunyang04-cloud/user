import { act, render, screen } from "@testing-library/react";
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
});
