import { useEffect, useState } from "react";
import { RefreshCw } from "lucide-react";
import type { DataSourcesPayload, ExecutionApi } from "../types";
import { DataTable, ErrorState, Field, LoadingState, PageHeader, Panel, Stat } from "../components";
import { text } from "../format";

interface DataPageProps {
  api: ExecutionApi;
}

export function DataPage({ api }: DataPageProps): JSX.Element {
  const [payload, setPayload] = useState<DataSourcesPayload | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [asOfDate, setAsOfDate] = useState("");
  const [startDate, setStartDate] = useState("");
  const [universe, setUniverse] = useState("all_a");
  const [domains, setDomains] = useState("market_daily");
  const [jobMessage, setJobMessage] = useState("");

  const load = (): void => {
    setLoading(true);
    api
      .getDataSources()
      .then((next) => {
        setPayload(next);
        const defaults = next.data_platform.default_refresh;
        if (defaults) {
          setAsOfDate(defaults.as_of_date || "");
          setUniverse(defaults.universe || "all_a");
          setDomains((defaults.domains || []).join(","));
        }
        setError("");
      })
      .catch((err: Error) => setError(err.message))
      .finally(() => setLoading(false));
  };

  useEffect(() => {
    load();
  }, []);

  async function refresh(): Promise<void> {
    setJobMessage("");
    try {
      const response = await api.refreshDataSources({
        as_of_date: asOfDate,
        start_date: startDate,
        universe,
        domains: domains.split(",").map((item) => item.trim()).filter(Boolean),
        force_unlock: false
      });
      setJobMessage(`已提交作业 ${response.job_id || ""} (${response.status})`);
    } catch (err) {
      setJobMessage(err instanceof Error ? err.message : "提交失败");
    }
  }

  return (
    <div>
      <PageHeader title="数据" eyebrow="数据湖、data platform refresh 与可用边界" actions={<button onClick={load}><RefreshCw size={16} />刷新</button>} />
      {loading ? <LoadingState /> : null}
      {error ? <ErrorState message={error} /> : null}
      <div className="stat-grid">
        <Stat label="Lake Root" value={text(payload?.lake_root)} />
        <Stat label="Catalog" value={text(payload?.catalog_status)} />
        <Stat label="Dataset 数" value={payload?.datasets.length || 0} />
        <Stat label="最新 Refresh" value={text(payload?.data_platform.latest_refresh_run)} />
      </div>
      <Panel title="显式刷新">
        <div className="form-grid">
          <Field label="As-of 日期">
            <input value={asOfDate} onChange={(event) => setAsOfDate(event.target.value)} placeholder="20260522" />
          </Field>
          <Field label="开始日期">
            <input value={startDate} onChange={(event) => setStartDate(event.target.value)} placeholder="可选" />
          </Field>
          <Field label="股票池">
            <input value={universe} onChange={(event) => setUniverse(event.target.value)} />
          </Field>
          <Field label="Domains">
            <input value={domains} onChange={(event) => setDomains(event.target.value)} />
          </Field>
        </div>
        <button className="primary" onClick={refresh}>刷新数据</button>
        {jobMessage ? <p className="inline-message">{jobMessage}</p> : null}
      </Panel>
      <Panel title="Lake Datasets">
        <DataTable
          rows={(payload?.datasets || []).map((row) => ({ ...row }))}
          preferredColumns={["dataset_id", "dataset_kind", "domain", "zone", "status", "start_date", "end_date", "created_at"]}
          emptyText="暂无 dataset"
        />
      </Panel>
    </div>
  );
}
