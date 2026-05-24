import { useEffect, useMemo, useState } from "react";
import { RefreshCw } from "lucide-react";
import type { ExecutionApi, StatusPayload, TableRow } from "../types";
import { DataTable, ErrorState, LoadingState, PageHeader, Panel, Stat, StatusPill } from "../components";
import { text } from "../format";

interface OverviewPageProps {
  api: ExecutionApi;
}

export function OverviewPage({ api }: OverviewPageProps): JSX.Element {
  const [payload, setPayload] = useState<StatusPayload | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  const load = (): void => {
    setLoading(true);
    api
      .getStatus(10)
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

  const manifest = payload?.active_manifest || {};
  const tradePlan = payload?.latest_trade_plan;
  const paper = payload?.paper_account || {};
  const jobs = useMemo<TableRow[]>(
    () =>
      (payload?.recent_jobs || []).map((job) => ({
        job_id: job.job_id,
        task_name: job.task_name,
        status: job.status,
        started_at: job.started_at,
        exit_code: job.exit_code
      })),
    [payload]
  );

  return (
    <div>
      <PageHeader title="总览" eyebrow="当前执行状态、模型、账户与最近作业" actions={<button onClick={load}><RefreshCw size={16} />刷新</button>} />
      {loading ? <LoadingState /> : null}
      {error ? <ErrorState message={error} /> : null}
      <div className="stat-grid">
        <Stat label="运行时" value={text(payload?.runtime_root)} />
        <Stat label="Active 策略" value={text(manifest.strategy_name || manifest.candidate_label)} />
        <Stat label="交易计划" value={<StatusPill value={tradePlan?.status || "missing"} />} />
        <Stat label="动作数" value={tradePlan?.actions?.length || 0} />
        <Stat label="账户权益" value={text(paper.latest_equity && typeof paper.latest_equity === "object" ? (paper.latest_equity as Record<string, unknown>).total_equity : "")} />
        <Stat label="未成交" value={text(paper.pending_order_count)} />
        <Stat label="账户持仓" value={text(payload?.current_positions?.row_count || paper.position_count)} />
        <Stat label="锁状态" value={<StatusPill value={payload?.lock && Object.keys(payload.lock).length ? "locked" : "ok"} />} />
      </div>
      <div className="two-column">
        <Panel title="Active Manifest 摘要">
          <div className="key-list">
            <span>路径</span><strong>{text(manifest.path)}</strong>
            <span>候选</span><strong>{text(manifest.candidate_label)}</strong>
            <span>股票池</span><strong>{text(manifest.liquidity_pool_name)}</strong>
            <span>执行 profile</span><strong>{text(manifest.effective_live_execution_profile)}</strong>
          </div>
        </Panel>
        <Panel title="最新交易计划">
          <div className="key-list">
            <span>信号日</span><strong>{text(tradePlan?.summary?.signal_date)}</strong>
            <span>现金</span><strong>{text(tradePlan?.summary?.cash_input)}</strong>
            <span>模型训练</span><strong>{text(tradePlan?.model_info?.trained_at)}</strong>
            <span>相差交易日</span><strong>{text(tradePlan?.model_info?.trading_day_lag)}</strong>
          </div>
        </Panel>
        <Panel title="模拟账户">
          <div className="key-list">
            <span>账本</span><strong>{text(paper.source)}</strong>
            <span>现金</span><strong>{text(paper.available_cash)}</strong>
            <span>权益日期</span><strong>{text(paper.latest_equity && typeof paper.latest_equity === "object" ? (paper.latest_equity as Record<string, unknown>).as_of_date : "")}</strong>
            <span>待执行订单</span><strong>{text(paper.pending_order_count)}</strong>
          </div>
        </Panel>
      </div>
      <Panel title="最近作业">
        <DataTable rows={jobs} preferredColumns={["job_id", "task_name", "status", "started_at", "exit_code"]} emptyText="暂无作业" />
      </Panel>
      {payload?.warnings?.length ? (
        <Panel title="系统提示">
          <ul className="compact-list">
            {payload.warnings.map((item) => <li key={item}>{item}</li>)}
          </ul>
        </Panel>
      ) : null}
    </div>
  );
}
