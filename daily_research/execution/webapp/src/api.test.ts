import { afterEach, describe, expect, it, vi } from "vitest";
import { createApiClient } from "./api";

describe("api client", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("loads model summaries from /api/models", async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({ status: "ok", models: [] })
    });
    const api = createApiClient(fetchMock as unknown as typeof fetch);

    await api.getModels();

    expect(fetchMock).toHaveBeenCalledWith("/api/models", expect.objectContaining({ method: "GET" }));
  });

  it("posts explicit model training requests", async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({ job_id: "job1", status: "queued" })
    });
    const api = createApiClient(fetchMock as unknown as typeof fetch);

    await api.trainModel("production-full-fit", {
      dataset_mode: "dataset_id",
      dataset_id: "policy_input_bundle__20260522",
      start_date: "20250101",
      end_date: "20260522",
      force_unlock: false
    });

    expect(fetchMock).toHaveBeenCalledWith(
      "/api/models/production-full-fit/train",
      expect.objectContaining({
        method: "POST",
        body: JSON.stringify({
          dataset_mode: "dataset_id",
          dataset_id: "policy_input_bundle__20260522",
          start_date: "20250101",
          end_date: "20260522",
          force_unlock: false
        })
      })
    );
  });
});
