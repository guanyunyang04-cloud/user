import { useEffect, useState } from "react";
import { Unlock } from "lucide-react";
import type { DoctorPayload, ExecutionApi, StatusPayload } from "../types";
import { DataTable, ErrorState, LoadingState, PageHeader, Panel, Stat, StatusPill } from "../components";
import { text } from "../format";

interface SystemPageProps {
  api: ExecutionApi;
}

export function SystemPage({ api }: SystemPageProps): JSX.Element {
  const [status, setStatus] = useState<StatusPayload | null>(null);
  const [doctor, setDoctor] = useState<DoctorPayload | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [unlockConfirm, setUnlockConfirm] = useState(false);
  const [message, setMessage] = useState("");

  const load = (): void => {
    setLoading(true);
    Promise.all([api.getStatus(20), api.getDoctor()])
      .then(([nextStatus, nextDoctor]) => {
        setStatus(nextStatus);
        setDoctor(nextDoctor);
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

  return (
    <div>
      <PageHeader title="系统" eyebrow="doctor、runtime、锁状态与环境信息" />
      {loading ? <LoadingState /> : null}
      {error ? <ErrorState message={error} /> : null}
      <div className="stat-grid">
        <Stat label="Doctor" value={<StatusPill value={doctor?.status || "unknown"} />} />
        <Stat label="Yolos Python" value={text(status?.yolos_python)} />
        <Stat label="Runtime Root" value={text(status?.runtime_root)} />
        <Stat label="锁" value={<StatusPill value={status?.lock && Object.keys(status.lock).length ? "locked" : "ok"} />} />
        <Stat label="Active Dataset" value={text(status?.active_manifest?.lake_dataset_id || status?.active_manifest?.source_market_dataset_id)} />
        <Stat label="交易计划" value={<StatusPill value={String(status?.latest_trade_plan?.status || "missing")} />} />
        <Stat label="自动更新" value={<StatusPill value={Boolean(status?.scheduler_status?.enabled)} />} />
        <Stat label="盘后检查" value={text(status?.scheduler_status?.post_close_time)} />
      </div>
      <Panel title="关键证据">
        <div className="key-list">
          <span>Active Manifest</span>
          <strong>{text(status?.active_manifest?.path)}</strong>
          <span>Data Source</span>
          <strong>{text(status?.active_manifest?.data_source)}</strong>
          <span>Lake Dataset</span>
          <strong>{text(status?.active_manifest?.lake_dataset_id || status?.active_manifest?.source_market_dataset_id)}</strong>
          <span>Dataset End Date</span>
          <strong>{text(status?.active_manifest?.lake_dataset_end_date)}</strong>
          <span>Trade Plan Run</span>
          <strong>{text(status?.latest_trade_plan?.artifact_paths?.run_dir)}</strong>
          <span>Trade Plan TXT</span>
          <strong>{text(status?.latest_trade_plan?.artifact_paths?.txt || status?.latest_trade_plan?.path)}</strong>
        </div>
      </Panel>
      <Panel title="自动盘后更新">
        <div className="key-list">
          <span>启用</span>
          <strong><StatusPill value={Boolean(status?.scheduler_status?.enabled)} /></strong>
          <span>盘后检查时间</span>
          <strong>{text(status?.scheduler_status?.post_close_time)}</strong>
          <span>时区</span>
          <strong>{text(status?.scheduler_status?.timezone)}</strong>
          <span>下一次检查</span>
          <strong>{text(status?.scheduler_status?.next_check_at)}</strong>
          <span>错过状态</span>
          <strong>{text(status?.scheduler_status?.missed_status)}</strong>
          <span>最近自动 Job</span>
          <strong>{text(status?.last_auto_refresh?.job_id || status?.scheduler_status?.last_auto_refresh?.job_id)}</strong>
          <span>最近决策</span>
          <strong>{text(status?.last_auto_refresh?.scheduler_decision || status?.scheduler_status?.recent_decision?.decision)}</strong>
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
      <Panel title="Runtime JSON">
        <pre className="detail-json">{JSON.stringify(status, null, 2)}</pre>
      </Panel>
    </div>
  );
}
