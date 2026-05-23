import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { TradePlanPage } from "./TradePlanPage";
import type { ExecutionApi } from "../types";

describe("TradePlanPage", () => {
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
});
