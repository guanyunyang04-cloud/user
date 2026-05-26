import { useEffect, useState } from "react";
import { Unlock } from "lucide-react";
import type { DailyRunStatusPayload, DoctorPayload, ExecutionApi, SchedulerPayload } from "../types";
import { DataTable, ErrorState, LoadingState, PageHeader, Panel, Stat, StatusPill } from "../components";
import { text } from "../format";

interface SystemPageProps {
  api: ExecutionApi;
}

export function SystemPage({ api }: SystemPageProps): JSX.Element {
  const [dailyRun, setDailyRun] = useState<DailyRunStatusPayload | null>(null);
  const [scheduler, setScheduler] = useState<SchedulerPayload | null>(null);
  const [doctor, setDoctor] = useState<DoctorPayload | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [unlockConfirm, setUnlockConfirm] = useState(false);
  const [message, setMessage] = useState("");

  const load = (): void => {
    setLoading(true);
    Promise.all([api.getDailyRunStatus(), api.getSystemDoctor(), api.getScheduler()])
      .then(([nextDailyRun, nextDoctor, nextScheduler]) => {
        setDailyRun(nextDailyRun);
        setDoctor(nextDoctor);
        setScheduler(nextScheduler);
        setError("");
      })
      .catch((err: Error) => setError(err.message))
      .finally(() => setLoading(false));
  };

  useEffect(() => {
    load();
  }, []);

  async function unlock(): Promise<void> {
    setMessage("");
    try {
      const response = await api.unlockRuntime(unlockConfirm);
      setMessage(response.detail);
      load();
    } catch (err) {
      setError(err instanceof Error ? err.message : "解锁失败");
    }
  }

  const verdict = dailyRun?.latest_verdict || {};

  return (
    <div>
      <PageHeader title="系统" eyebrow="doctor、系统任务与 daily verdict" />
      {loading ? <LoadingState /> : null}
      {error ? <ErrorState message={error} /> : null}
      <div className="stat-grid">
        <Stat label="Doctor" value={<StatusPill value={doctor?.status || "unknown"} />} />
        <Stat label="Daily Run" value={<StatusPill value={dailyRun?.status || "missing"} />} />
        <Stat label="目标交易日" value={text(verdict.target_trading_date)} />
        <Stat label="阻断原因" value={text(verdict.blocker_code)} />
        <Stat label="系统任务" value={<StatusPill value={Boolean(scheduler?.enabled)} />} />
        <Stat label="任务名" value={text(scheduler?.task_name)} />
        <Stat label="下次运行" value={text(scheduler?.next_run_time)} />
        <Stat label="上次结果" value={text(scheduler?.last_result)} />
      </div>
      <Panel title="关键证据">
        <div className="key-list">
          <span>Daily Runs Root</span>
          <strong>{text(dailyRun?.daily_runs_root)}</strong>
          <span>Verdict</span>
          <strong>{text(verdict.evidence_paths?.verdict)}</strong>
          <span>Dataset</span>
          <strong>{text(verdict.dataset_id)}</strong>
          <span>Signal Panel Date</span>
          <strong>{text(verdict.signal_panel_date)}</strong>
          <span>Trade Plan Run</span>
          <strong>{text(verdict.trade_plan_run_dir)}</strong>
          <span>Paper Reconcile</span>
          <strong>{text(verdict.paper_reconcile_status)}</strong>
        </div>
      </Panel>
      <Panel title="Windows Task Scheduler">
        <div className="key-list">
          <span>Installed</span>
          <strong><StatusPill value={Boolean(scheduler?.installed)} /></strong>
          <span>Enabled</span>
          <strong><StatusPill value={Boolean(scheduler?.enabled)} /></strong>
          <span>Task Name</span>
          <strong>{text(scheduler?.task_name)}</strong>
          <span>Next Run</span>
          <strong>{text(scheduler?.next_run_time)}</strong>
          <span>Last Run</span>
          <strong>{text(scheduler?.last_run_time)}</strong>
          <span>Last Result</span>
          <strong>{text(scheduler?.last_result)}</strong>
        </div>
      </Panel>
      <Panel title="Doctor Checks">
        <DataTable
          rows={(doctor?.checks || []).map((check) => ({ name: check.name, ok: String(check.ok), detail: check.detail }))}
          preferredColumns={["name", "ok", "detail"]}
          emptyText="暂无检查"
        />
      </Panel>
      <Panel title="锁控制">
        <label className="confirm-line">
          <input type="checkbox" checked={unlockConfirm} onChange={(event) => setUnlockConfirm(event.target.checked)} />
          确认 force unlock
        </label>
        <button onClick={unlock}><Unlock size={16} />清理锁</button>
        {message ? <p className="inline-message">{message}</p> : null}
      </Panel>
    </div>
  );
}
