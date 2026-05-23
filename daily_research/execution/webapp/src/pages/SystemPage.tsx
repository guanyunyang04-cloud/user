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
      </div>
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
