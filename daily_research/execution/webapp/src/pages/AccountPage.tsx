import { useEffect, useState } from "react";
import { Plus, RefreshCw, RotateCcw, Save, Trash2, WalletCards } from "lucide-react";
import type { AccountPosition, ExecutionApi, PaperAccountPayload, PaperPerformancePayload, TableRow } from "../types";
import { DataTable, ErrorState, Field, LoadingState, PageHeader, Panel, Stat, StatusPill } from "../components";
import { text } from "../format";

interface AccountPageProps {
  api: ExecutionApi;
}

function rows(value: unknown): TableRow[] {
  return Array.isArray(value) ? (value as TableRow[]) : [];
}

export function AccountPage({ api }: AccountPageProps): JSX.Element {
  const [payload, setPayload] = useState<PaperAccountPayload | null>(null);
  const [performance, setPerformance] = useState<PaperPerformancePayload | null>(null);
  const [cash, setCash] = useState("");
  const [positions, setPositions] = useState<AccountPosition[]>([]);
  const [flowType, setFlowType] = useState("deposit");
  const [flowAmount, setFlowAmount] = useState("");
  const [flowReason, setFlowReason] = useState("");
  const [manualStock, setManualStock] = useState("");
  const [manualShares, setManualShares] = useState("");
  const [manualCost, setManualCost] = useState("");
  const [periodStart, setPeriodStart] = useState("");
  const [periodEnd, setPeriodEnd] = useState("");
  const [loading, setLoading] = useState(true);
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");

  const load = (): void => {
    setLoading(true);
    api
      .getPaperAccount()
      .then((next) => {
        setPayload(next);
        setCash(next.available_cash === null ? "" : String(next.available_cash));
        setPositions(next.positions || []);
        const equityDate = String(next.latest_equity?.as_of_date || "");
        if (equityDate) {
          setPeriodEnd((current) => current || equityDate);
        }
        setError("");
      })
      .catch((err: Error) => setError(err.message))
      .finally(() => setLoading(false));
  };

  useEffect(() => {
    load();
  }, []);

  function updatePosition(index: number, patch: Partial<AccountPosition>): void {
    setPositions((current) => current.map((item, itemIndex) => (itemIndex === index ? { ...item, ...patch } : item)));
  }

  async function saveSnapshot(): Promise<void> {
    setMessage("");
    try {
      const next = await api.saveAccount({ available_cash: cash, positions });
      setPayload(next);
      setMessage("手动快照修正已写入账本");
    } catch (err) {
      setError(err instanceof Error ? err.message : "保存失败");
    }
  }

  async function reset(): Promise<void> {
    setMessage("");
    try {
      await api.resetAccountExample();
      load();
      setMessage("已从示例账户恢复并写入账本");
    } catch (err) {
      setError(err instanceof Error ? err.message : "恢复失败");
    }
  }

  async function submitCashFlow(): Promise<void> {
    setMessage("");
    try {
      const next = await api.recordPaperCashFlow({ flow_type: flowType, amount: flowAmount, reason: flowReason });
      setPayload(next);
      setCash(next.available_cash === null ? "" : String(next.available_cash));
      setFlowAmount("");
      setFlowReason("");
      setMessage("现金流水已记录");
    } catch (err) {
      setError(err instanceof Error ? err.message : "现金流水提交失败");
    }
  }

  async function submitPositionAdjustment(): Promise<void> {
    setMessage("");
    try {
      const next = await api.recordPaperManualAdjustment({
        adjustment_type: "position",
        stock: manualStock,
        shares: manualShares,
        cost_price: manualCost,
        reason: "manual position correction"
      });
      setPayload(next);
      setPositions(next.positions || []);
      setManualStock("");
      setManualShares("");
      setManualCost("");
      setMessage("持仓修正已记录");
    } catch (err) {
      setError(err instanceof Error ? err.message : "持仓修正失败");
    }
  }

  async function applyLatestPlan(): Promise<void> {
    setMessage("");
    try {
      const result = await api.applyLatestPaperPlan({});
      setMessage(`模拟过账：${text(result.status)} ${text((result.apply_result as Record<string, unknown> | undefined)?.filled_order_count)} 笔成交`);
      load();
    } catch (err) {
      setError(err instanceof Error ? err.message : "模拟过账失败");
    }
  }

  async function queryPerformance(): Promise<void> {
    setMessage("");
    try {
      const next = await api.getPaperPerformance(periodStart, periodEnd);
      setPerformance(next);
      setMessage("区间收益已更新");
    } catch (err) {
      setError(err instanceof Error ? err.message : "收益查询失败");
    }
  }

  const latestEquity = payload?.latest_equity || {};

  return (
    <div>
      <PageHeader
        title="模拟账户"
        eyebrow="本地 paper ledger、订单、成交、现金流水与区间收益"
        actions={<button onClick={load}><RefreshCw size={16} />刷新</button>}
      />
      {loading ? <LoadingState /> : null}
      {error ? <ErrorState message={error} /> : null}
      <div className="stat-grid">
        <Stat label="账本" value={<StatusPill value={payload?.source || "paper_ledger"} />} />
        <Stat label="现金" value={text(payload?.available_cash)} />
        <Stat label="总权益" value={text(latestEquity.total_equity)} />
        <Stat label="权益日期" value={text(latestEquity.as_of_date)} />
        <Stat label="持仓数" value={payload?.position_count || 0} />
        <Stat label="未成交" value={payload?.pending_order_count || 0} />
        <Stat label="已成交" value={payload?.filled_order_count || 0} />
        <Stat label="阻塞订单" value={payload?.blocked_order_count || 0} />
      </div>

      <div className="two-column">
        <Panel title="现金流水">
          <div className="form-grid">
            <Field label="类型">
              <select value={flowType} onChange={(event) => setFlowType(event.target.value)}>
                <option value="deposit">充值</option>
                <option value="withdrawal">提现</option>
              </select>
            </Field>
            <Field label="金额">
              <input value={flowAmount} onChange={(event) => setFlowAmount(event.target.value)} />
            </Field>
            <Field label="原因">
              <input value={flowReason} onChange={(event) => setFlowReason(event.target.value)} />
            </Field>
          </div>
          <button className="primary" onClick={submitCashFlow}><WalletCards size={16} />提交现金流水</button>
        </Panel>

        <Panel title="区间收益">
          <div className="form-grid">
            <Field label="开始日期">
              <input value={periodStart} onChange={(event) => setPeriodStart(event.target.value)} placeholder="2026-05-22" />
            </Field>
            <Field label="结束日期">
              <input value={periodEnd} onChange={(event) => setPeriodEnd(event.target.value)} placeholder="2026-05-25" />
            </Field>
          </div>
          <button onClick={queryPerformance}>查询收益</button>
          <div className="key-list compact-metrics">
            <span>TWR</span><strong>{text(performance?.total_return)}</strong>
            <span>最大回撤</span><strong>{text(performance?.max_drawdown)}</strong>
            <span>净现金流</span><strong>{text(performance?.net_cash_flow)}</strong>
          </div>
        </Panel>
      </div>

      <Panel title="交易计划过账">
        <button className="primary" onClick={applyLatestPlan}>按最新交易计划模拟过账</button>
        {message ? <p className="inline-message">{message}</p> : null}
      </Panel>

      <Panel title="持仓">
        <DataTable rows={rows(payload?.positions)} preferredColumns={["stock", "shares", "cost_price"]} />
      </Panel>

      <Panel title="未成交订单">
        <DataTable rows={rows(payload?.pending_orders)} preferredColumns={["stock", "side", "remaining_shares", "execution_date", "status", "reason"]} />
      </Panel>

      <Panel title="成交流水">
        <DataTable rows={rows(payload?.recent_fills)} preferredColumns={["stock", "side", "shares", "price", "execution_date"]} />
      </Panel>

      <Panel title="现金流水记录">
        <DataTable rows={rows(payload?.recent_cash_flows)} preferredColumns={["flow_date", "flow_type", "amount", "reason"]} />
      </Panel>

      <Panel title="手动持仓修正">
        <div className="form-grid">
          <Field label="证券代码">
            <input value={manualStock} onChange={(event) => setManualStock(event.target.value)} />
          </Field>
          <Field label="数量">
            <input value={manualShares} onChange={(event) => setManualShares(event.target.value)} />
          </Field>
          <Field label="成本价">
            <input value={manualCost} onChange={(event) => setManualCost(event.target.value)} />
          </Field>
        </div>
        <button onClick={submitPositionAdjustment}>提交持仓修正</button>
      </Panel>

      <Panel title="兼容快照修正">
        <Field label="可用现金">
          <input value={cash} onChange={(event) => setCash(event.target.value)} />
        </Field>
        <div className="positions-editor">
          <div className="positions-head">
            <span>证券代码</span>
            <span>数量</span>
            <span>成本价</span>
            <span />
          </div>
          {positions.map((position, index) => (
            <div className="positions-row" key={`${position.stock}-${index}`}>
              <input value={position.stock} onChange={(event) => updatePosition(index, { stock: event.target.value })} />
              <input value={position.shares} onChange={(event) => updatePosition(index, { shares: event.target.value })} />
              <input value={position.cost_price} onChange={(event) => updatePosition(index, { cost_price: event.target.value })} />
              <button onClick={() => setPositions((current) => current.filter((_, itemIndex) => itemIndex !== index))} aria-label={`删除 ${position.stock || index}`}>
                <Trash2 size={15} />
              </button>
            </div>
          ))}
        </div>
        <div className="button-row">
          <button onClick={() => setPositions((current) => [...current, { stock: "", shares: "", cost_price: "" }])}><Plus size={16} />添加持仓</button>
          <button onClick={reset}><RotateCcw size={16} />示例恢复</button>
          <button className="primary" onClick={saveSnapshot}><Save size={16} />保存修正</button>
        </div>
      </Panel>
    </div>
  );
}
