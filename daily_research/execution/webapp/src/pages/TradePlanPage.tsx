import { useEffect, useMemo, useState } from "react";
import { Play, RefreshCw } from "lucide-react";
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { ExecutionApi, TradePlanPayload } from "../types";
import { DataTable, EmptyState, ErrorState, Field, LoadingState, PageHeader, Panel, Stat } from "../components";
import { numberValue, pick, text } from "../format";

interface TradePlanPageProps {
  api: ExecutionApi;
}

export function TradePlanPage({ api }: TradePlanPageProps): JSX.Element {
  const [payload, setPayload] = useState<TradePlanPayload | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [candidateProfile, setCandidateProfile] = useState("active_execution_strategy");
  const [cash, setCash] = useState("");
  const [jobMessage, setJobMessage] = useState("");

  const load = (): void => {
    setLoading(true);
    api
      .getTradePlan()
      .then((next) => {
        setPayload(next);
        setError("");
      })
      .catch((err: Error) => setError(err.message))
      .finally(() => setLoading(false));
  };

  useEffect(() => {
    load();
  }, []);

  const chartRows = useMemo(
    () =>
      (payload?.holdings || []).slice(0, 12).map((row) => ({
        stock: text(pick(row, ["stock", "code", "symbol"])),
        current: numberValue(pick(row, ["current_weight", "weight", "current_position_weight"])),
        target: numberValue(pick(row, ["target_weight", "target_position_weight"]))
      })),
    [payload]
  );

  async function generate(): Promise<void> {
    setJobMessage("");
    try {
      const response = await api.generateTradePlan({
        candidate_profile: candidateProfile,
        cash,
        force_unlock: false
      });
      setJobMessage(`已提交作业 ${response.job_id || ""} (${response.status})`);
    } catch (err) {
      setJobMessage(err instanceof Error ? err.message : "提交失败");
    }
  }

  const summary = payload?.summary || {};
  const missing = payload?.status === "missing" || payload?.exists === false;

  return (
    <div>
      <PageHeader
        title="交易计划"
        eyebrow="结构化动作、持仓、观察池与 TXT 原文"
        actions={<button onClick={load}><RefreshCw size={16} />刷新</button>}
      />
      {loading ? <LoadingState /> : null}
      {error ? <ErrorState message={error} /> : null}
      {missing ? <EmptyState title="交易计划未生成" detail="缺少 latest_trade_plan.txt 和结构化 run 产物时会显示此状态。" /> : null}

      <Panel title="生成交易计划">
        <div className="form-grid">
          <Field label="候选配置">
            <input value={candidateProfile} onChange={(event) => setCandidateProfile(event.target.value)} />
          </Field>
          <Field label="现金覆盖">
            <input value={cash} onChange={(event) => setCash(event.target.value)} placeholder="留空使用账户文件" />
          </Field>
        </div>
        <button className="primary" onClick={generate}><Play size={16} />生成交易计划</button>
        {jobMessage ? <p className="inline-message">{jobMessage}</p> : null}
      </Panel>

      <div className="stat-grid">
        <Stat label="信号日" value={text(summary.signal_date)} />
        <Stat label="输入现金" value={text(summary.cash_input)} />
        <Stat label="计划后现金" value={text(summary.estimated_cash_after_plan)} />
        <Stat label="动作数" value={payload?.actions.length || 0} />
        <Stat label="模型训练时间" value={text(payload?.model_info?.trained_at)} />
        <Stat label="交易日间隔" value={text(payload?.model_info?.trading_day_lag)} />
      </div>

      <div className="two-column">
        <Panel title="目标权重对比">
          {chartRows.length ? (
            <div className="chart-box">
              <ResponsiveContainer width="100%" height={260}>
                <BarChart data={chartRows}>
                  <CartesianGrid strokeDasharray="3 3" vertical={false} />
                  <XAxis dataKey="stock" tick={{ fontSize: 11 }} />
                  <YAxis tick={{ fontSize: 11 }} />
                  <Tooltip />
                  <Bar dataKey="current" fill="#64748b" name="当前" />
                  <Bar dataKey="target" fill="#2563eb" name="目标" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          ) : (
            <EmptyState title="暂无持仓权重图" />
          )}
        </Panel>
        <Panel title="模型信息">
          <div className="key-list">
            <span>训练时间</span><strong>{text(payload?.model_info?.trained_at)}</strong>
            <span>样本截止</span><strong>{text(payload?.model_info?.artifact_latest_data_date || payload?.model_info?.train_end_date)}</strong>
            <span>最新完成交易日</span><strong>{text(payload?.model_info?.latest_completed_trading_date)}</strong>
            <span>相差交易日数</span><strong>{text(payload?.model_info?.trading_day_lag)}</strong>
          </div>
        </Panel>
      </div>

      <Panel title="今日动作">
        <DataTable rows={payload?.actions || []} preferredColumns={["stock", "action", "shares_delta", "target_weight", "order_value"]} />
      </Panel>
      <Panel title="持仓快照">
        <DataTable rows={payload?.holdings || []} preferredColumns={["stock", "shares", "current_weight", "target_weight", "market_value"]} />
      </Panel>
      <Panel title="Watchlist">
        <DataTable rows={payload?.watchlist || []} preferredColumns={["stock", "score", "target_weight", "rank"]} />
      </Panel>
      <Panel title="TXT 原文">
        <pre className="txt-preview">{payload?.txt_preview?.join("\n") || "未生成"}</pre>
      </Panel>
    </div>
  );
}
