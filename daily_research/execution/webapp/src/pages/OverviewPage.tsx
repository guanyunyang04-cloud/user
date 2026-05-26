import { useEffect, useMemo, useState } from "react";
import { RefreshCw } from "lucide-react";
import type { DailyRunStatusPayload, ExecutionApi, JobSummary, PaperAccountPayload, TableRow, TradePlanPayload } from "../types";
import { DataTable, ErrorState, LoadingState, PageHeader, Panel, Stat, StatusPill } from "../components";
import { text } from "../format";

interface OverviewPageProps {
  api: ExecutionApi;
}

export function OverviewPage({ api }: OverviewPageProps): JSX.Element {
  const [dailyRun, setDailyRun] = useState<DailyRunStatusPayload | null>(null);
  const [tradePlan, setTradePlan] = useState<TradePlanPayload | null>(null);
  const [paper, setPaper] = useState<PaperAccountPayload | null>(null);
  const [recentJobs, setRecentJobs] = useState<JobSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  const load = (): void => {
    setLoading(true);
    Promise.all([api.getDailyRunStatus(), api.getTradePlan(), api.getPaperAccount(), api.getJobs(10)])
      .then(([nextDailyRun, nextTradePlan, nextPaper, nextJobs]) => {
        setDailyRun(nextDailyRun);
        setTradePlan(nextTradePlan);
        setPaper(nextPaper);
        setRecentJobs(nextJobs);
        setError("");
      })
      .catch((err: Error) => setError(err.message))
      .finally(() => setLoading(false));
  };

  useEffect(() => {
    load();
  }, []);

  const verdict = dailyRun?.latest_verdict || {};
  const jobs = useMemo<TableRow[]>(
    () =>
      recentJobs.map((job) => ({
        job_id: job.job_id,
        task_name: job.task_name,
        status: job.status,
        started_at: job.started_at,
        exit_code: job.exit_code
      })),
    [recentJobs]
  );

  return (
    <div>
      <PageHeader title="总览" eyebrow="每日计划 verdict、账户与最近作业" actions={<button onClick={load}><RefreshCw size={16} />刷新</button>} />
      {loading ? <LoadingState /> : null}
      {error ? <ErrorState message={error} /> : null}
      <div className="stat-grid">
        <Stat label="每日计划" value={<StatusPill value={dailyRun?.status || "missing"} />} />
        <Stat label="目标交易日" value={text(verdict.target_trading_date)} />
        <Stat label="阻断原因" value={text(verdict.blocker_code)} />
        <Stat label="交易计划" value={<StatusPill value={tradePlan?.status || "missing"} />} />
        <Stat label="动作数" value={tradePlan?.actions?.length || 0} />
        <Stat label="账户权益" value={text(paper?.latest_equity && typeof paper.latest_equity === "object" ? (paper.latest_equity as Record<string, unknown>).total_equity : "")} />
        <Stat label="未成交" value={text(paper?.pending_order_count)} />
        <Stat label="账户持仓" value={text(paper?.position_count)} />
        <Stat label="执行方式" value={<StatusPill value="manual" />} />
      </div>
      <div className="two-column">
        <Panel title="Daily Verdict">
          <div className="key-list">
            <span>Run Date</span><strong>{text(verdict.run_date)}</strong>
            <span>目标交易日</span><strong>{text(verdict.target_trading_date)}</strong>
            <span>Dataset</span><strong>{text(verdict.dataset_id)}</strong>
            <span>Signal Panel</span><strong>{text(verdict.signal_panel_date)}</strong>
            <span>Trade Plan Run</span><strong>{text(verdict.trade_plan_run_dir)}</strong>
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
            <span>账本</span><strong>{text(paper?.source)}</strong>
            <span>现金</span><strong>{text(paper?.available_cash)}</strong>
            <span>权益日期</span><strong>{text(paper?.latest_equity && typeof paper.latest_equity === "object" ? (paper.latest_equity as Record<string, unknown>).as_of_date : "")}</strong>
            <span>待执行订单</span><strong>{text(paper?.pending_order_count)}</strong>
          </div>
        </Panel>
      </div>
      <Panel title="最近作业">
        <DataTable rows={jobs} preferredColumns={["job_id", "task_name", "status", "started_at", "exit_code"]} emptyText="暂无作业" />
      </Panel>
    </div>
  );
}
