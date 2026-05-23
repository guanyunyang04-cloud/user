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
});
