import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { AccountPage } from "./AccountPage";
import type { ExecutionApi } from "../types";

function paperPayload() {
  return {
    status: "ok",
    source: "paper_ledger",
    path: "current_positions.csv",
    db_path: "paper.sqlite3",
    exists: true,
    available_cash: 100000,
    positions: [{ stock: "600000.SH", shares: 100, cost_price: 10.5 }],
    position_count: 1,
    total_shares: 100,
    last_modified_at: "2026-05-24T10:00:00",
    pending_order_count: 1,
    filled_order_count: 2,
    blocked_order_count: 0,
    pending_orders: [{ stock: "002866.SZ", side: "buy", remaining_shares: 1500, execution_date: "2026-05-25", status: "pending" }],
    recent_fills: [{ stock: "600000.SH", side: "sell", shares: 100, price: 11, execution_date: "2026-05-25" }],
    recent_cash_flows: [{ flow_type: "deposit", amount: 100000, reason: "initial", flow_date: "2026-05-22" }],
    latest_equity: { as_of_date: "2026-05-25", total_equity: 101000, daily_return: 0.01 }
  };
}

describe("AccountPage", () => {
  it("renders the paper trading workstation and records cash flow", async () => {
    const user = userEvent.setup();
    const getPaperAccount = vi.fn().mockResolvedValue(paperPayload());
    const recordPaperCashFlow = vi.fn().mockResolvedValue({ ...paperPayload(), available_cash: 150000 });
    const api = {
      getPaperAccount,
      recordPaperCashFlow,
      getPaperPerformance: vi.fn().mockResolvedValue({ status: "ok", total_return: 0, net_cash_flow: 0, points: [] }),
      applyLatestPaperPlan: vi.fn(),
      saveAccount: vi.fn(),
      resetAccountExample: vi.fn()
    } as unknown as ExecutionApi;

    render(<AccountPage api={api} />);

    expect(await screen.findByText("模拟账户")).toBeInTheDocument();
    expect(screen.getByText("paper_ledger")).toBeInTheDocument();
    expect(screen.getByText("002866.SZ")).toBeInTheDocument();
    expect(screen.getByText("成交流水")).toBeInTheDocument();

    await user.type(screen.getByLabelText("金额"), "50000");
    await user.type(screen.getByLabelText("原因"), "top up");
    await user.click(screen.getByRole("button", { name: "提交现金流水" }));

    await waitFor(() => {
      expect(recordPaperCashFlow).toHaveBeenCalledWith({ flow_type: "deposit", amount: "50000", reason: "top up" });
    });
  });
});
