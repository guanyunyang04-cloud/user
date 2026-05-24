import { useEffect, useState } from "react";
import { RefreshCw } from "lucide-react";
import type { DataSourcesPayload, ExecutionApi, JobDetail, ProviderHealthPayload, TableRow } from "../types";
import { DataTable, ErrorState, Field, LoadingState, PageHeader, Panel, Stat, StatusPill } from "../components";
import { text } from "../format";

interface DataPageProps {
  api: ExecutionApi;
  pollMs?: number;
}

function field(record: Record<string, unknown> | undefined, key: string): string {
  return text(record?.[key]);
}

function domainsFromText(value: string): string[] {
  return value.split(",").map((item) => item.trim()).filter(Boolean);
}

function summaryRows(summary: Record<string, unknown> | undefined): TableRow[] {
  return Object.entries(summary || {}).map(([key, value]) => ({ key, value: text(value) }));
}

export function DataPage({ api, pollMs = 3000 }: DataPageProps): JSX.Element {
  const [payload, setPayload] = useState<DataSourcesPayload | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [asOfDate, setAsOfDate] = useState("");
  const [startDate, setStartDate] = useState("");
  const [universe, setUniverse] = useState("all_a");
  const [domains, setDomains] = useState("market_daily");
  const [jobMessage, setJobMessage] = useState("");
  const [activeJobId, setActiveJobId] = useState("");
  const [jobDetail, setJobDetail] = useState<JobDetail | null>(null);
  const [jobError, setJobError] = useState("");
  const [healthChecking, setHealthChecking] = useState(false);
  const [providerHealth, setProviderHealth] = useState<ProviderHealthPayload | null>(null);
  const [providerHealthMessage, setProviderHealthMessage] = useState("");
  const signalNeedsRefresh = payload?.next_signal_action === "refresh";
  const refreshIsSkip = payload?.next_refresh_action === "skip" && !signalNeedsRefresh;
  const refreshButtonLabel = signalNeedsRefresh ? "刷新信号面板" : refreshIsSkip ? "已是最新" : "补齐到最新交易日";
  const providerPlan = payload?.formal_provider_plan || payload?.data_platform.default_refresh?.provider_plan || payload?.data_platform.provider_plan;

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
    if (refreshIsSkip) {
      setJobMessage(payload?.refresh_explanation || "当前数据集已是最新。");
      return;
    }
    setJobMessage("");
    setJobDetail(null);
    setJobError("");
    try {
      const response = await api.refreshDataSources({
        as_of_date: asOfDate,
        start_date: startDate,
        universe,
        domains: domainsFromText(domains),
        force_unlock: false
      });
      setJobMessage(`已提交作业 ${response.job_id || ""} (${response.status})`);
      setActiveJobId(response.job_id || "");
    } catch (err) {
      setJobMessage(err instanceof Error ? err.message : "提交失败");
    }
  }

  async function checkProviderHealth(): Promise<void> {
    setHealthChecking(true);
    setProviderHealthMessage("");
    try {
      const response = await api.runProviderHealth({
        as_of_date: asOfDate,
        domains: domainsFromText(domains),
        provider_plan: providerPlan || "formal_free_v3"
      });
      setProviderHealth(response);
      setProviderHealthMessage(`provider health: ${response.status}`);
    } catch (err) {
      setProviderHealthMessage(err instanceof Error ? err.message : "数据源检查失败");
    } finally {
      setHealthChecking(false);
    }
  }

  useEffect(() => {
    if (!activeJobId) {
      return undefined;
    }
    let disposed = false;
    let timer: number | undefined;
    const loadJob = (): void => {
      api
        .getJob(activeJobId, 160)
        .then((next) => {
          if (disposed) {
            return;
          }
          setJobDetail(next);
          setJobError("");
          if (["queued", "running"].includes(String(next.status || "").toLowerCase())) {
            timer = window.setTimeout(loadJob, pollMs);
          } else {
            load();
          }
        })
        .catch((err: Error) => {
          if (!disposed) {
            setJobError(err.message);
          }
        });
    };
    loadJob();
    return () => {
      disposed = true;
      if (timer !== undefined) {
        window.clearTimeout(timer);
      }
    };
  }, [api, activeJobId, pollMs]);

  return (
    <div>
      <PageHeader title="数据" eyebrow="数据湖、data platform refresh 与可用边界" actions={<button onClick={load}><RefreshCw size={16} />刷新</button>} />
      {loading ? <LoadingState /> : null}
      {error ? <ErrorState message={error} /> : null}
      <div className="stat-grid">
        <Stat label="Lake Root" value={text(payload?.lake_root)} />
        <Stat label="Catalog" value={text(payload?.catalog_status)} />
        <Stat label="Provider Plan" value={text(providerPlan)} />
        <Stat label="当前 End Date" value={text(payload?.active_dataset_end_date || payload?.latest_policy_input_dataset_end_date)} />
        <Stat label="最新完成交易日" value={text(payload?.data_platform.latest_completed_trading_date)} />
        <Stat label="Active Dataset" value={text(payload?.active_dataset_id)} />
        <Stat label="同步状态" value={<StatusPill value={payload?.dataset_sync_status || "unknown"} />} />
        <Stat label="刷新动作" value={<StatusPill value={payload?.next_refresh_action || "unknown"} />} />
        <Stat label="信号面板" value={<StatusPill value={payload?.signal_panel_status || "unknown"} />} />
        <Stat label="信号最新日" value={text(payload?.signal_panel_latest_date)} />
        <Stat label="Production Anchor" value={<StatusPill value={payload?.production_anchor_status || "unknown"} />} />
      </div>
      <Panel title="当前数据集">
        <div className="key-list">
          <span>最新完成交易日</span>
          <strong>{text(payload?.data_platform.latest_completed_trading_date)}</strong>
          <span>Active Dataset</span>
          <strong>{text(payload?.active_dataset_id)}</strong>
          <span>Active End Date</span>
          <strong>{text(payload?.active_dataset_end_date)}</strong>
          <span>Latest Policy Input</span>
          <strong>{text(payload?.latest_policy_input_dataset_id)}</strong>
          <span>Latest End Date</span>
          <strong>{text(payload?.latest_policy_input_dataset_end_date)}</strong>
          <span>当前状态</span>
          <strong><StatusPill value={payload?.current_dataset_status || "unknown"} /></strong>
          <span>是否最新</span>
          <strong><StatusPill value={payload?.is_current_dataset_latest} /></strong>
          <span>是否完整</span>
          <strong><StatusPill value={payload?.is_current_dataset_complete} /></strong>
          <span>刷新判断</span>
          <strong>{text(payload?.refresh_explanation)}</strong>
          <span>Refresh Manifest</span>
          <strong>{text(payload?.data_platform.latest_refresh_manifest_path)}</strong>
          <span>Refresh Status</span>
          <strong><StatusPill value={payload?.data_platform.latest_refresh_manifest_status || "unknown"} /></strong>
          <span>Registered Dataset</span>
          <strong>{text(payload?.data_platform.latest_refresh_registered_dataset_id)}</strong>
        </div>
        {(payload?.data_platform.latest_refresh_blockers || []).length ? (
          <ul className="compact-list">
            {(payload?.data_platform.latest_refresh_blockers || []).map((item) => (
              <li key={item}>{item}</li>
            ))}
          </ul>
        ) : null}
      </Panel>
      <div className="two-column">
        <Panel title="数据源健康">
          <div className="key-list">
            <span>正式方案</span>
            <strong>{text(providerPlan)}</strong>
            <span>最近检查</span>
            <strong><StatusPill value={providerHealth?.status || String(payload?.provider_health?.status || "unknown")} /></strong>
            <span>检查日期</span>
            <strong>{text(providerHealth?.as_of_date || payload?.data_platform.default_refresh?.as_of_date)}</strong>
            <span>Required Domains</span>
            <strong>{text(payload?.data_platform.default_refresh?.required_domains?.join(","))}</strong>
          </div>
          <button onClick={checkProviderHealth} disabled={healthChecking}>
            <RefreshCw size={16} />
            检查数据源
          </button>
          {providerHealthMessage ? <p className="inline-message">{providerHealthMessage}</p> : null}
          <DataTable
            rows={summaryRows(providerHealth?.summary || (payload?.provider_health?.summary as Record<string, unknown> | undefined))}
            preferredColumns={["key", "value"]}
            emptyText="暂无健康摘要"
          />
        </Panel>
        <Panel title="自动更新">
          <div className="key-list">
            <span>启用</span>
            <strong><StatusPill value={Boolean(payload?.scheduler_status?.enabled)} /></strong>
            <span>盘后检查</span>
            <strong>{text(payload?.scheduler_status?.post_close_time)}</strong>
            <span>时区</span>
            <strong>{text(payload?.scheduler_status?.timezone)}</strong>
            <span>下一次检查</span>
            <strong>{text(payload?.scheduler_status?.next_check_at)}</strong>
            <span>错过状态</span>
            <strong>{text(payload?.scheduler_status?.missed_status)}</strong>
            <span>最近自动作业</span>
            <strong>{text(payload?.last_auto_refresh?.job_id || payload?.scheduler_status?.last_auto_refresh?.job_id)}</strong>
          </div>
        </Panel>
      </div>
      <Panel title="Provider Matrix">
        <DataTable
          rows={(payload?.domain_matrix || []).map((row) => ({
            provider: row.provider,
            domain: row.domain,
            requirement: text(row.requirement),
            supported: String(Boolean(row.supported)),
            requires_token: String(Boolean(row.requires_token)),
            formal_refresh: String(Boolean(row.formal_refresh))
          }))}
          preferredColumns={["provider", "domain", "requirement", "supported", "requires_token", "formal_refresh"]}
          emptyText="暂无 provider matrix"
        />
      </Panel>
      <div className="two-column">
        <Panel title="生产信号面板">
          <div className="key-list">
            <span>状态</span>
            <strong><StatusPill value={payload?.signal_panel_status || "unknown"} /></strong>
            <span>最新完成交易日</span>
            <strong>{text(payload?.data_platform.latest_completed_trading_date)}</strong>
            <span>要求覆盖日期</span>
            <strong>{field(payload?.signal_panels, "required_date")}</strong>
            <span>面板最新日期</span>
            <strong>{text(payload?.signal_panel_latest_date)}</strong>
            <span>Target Panel 最新</span>
            <strong>{text(payload?.signal_panel_target_latest_date || field(payload?.signal_panels, "target_panel_latest_date"))}</strong>
            <span>Score Panel 最新</span>
            <strong>{text(payload?.signal_panel_score_latest_date || field(payload?.signal_panels, "score_panel_latest_date"))}</strong>
            <span>目标持仓数</span>
            <strong>{field(payload?.signal_panels, "target_position_count")}</strong>
            <span>下一步</span>
            <strong><StatusPill value={payload?.next_signal_action || field(payload?.signal_panels, "next_signal_action")} /></strong>
            <span>Target Panel</span>
            <strong>{field(payload?.signal_panels, "target_panel")}</strong>
            <span>Score Panel</span>
            <strong>{field(payload?.signal_panels, "score_panel")}</strong>
          </div>
        </Panel>
        <Panel title="Production 锚点">
          <div className="key-list">
            <span>状态</span>
            <strong><StatusPill value={payload?.production_anchor_status || "unknown"} /></strong>
            <span>Production Root</span>
            <strong>{field(payload?.production_anchor, "production_root")}</strong>
            <span>Full-fit Run</span>
            <strong>{field(payload?.production_anchor, "fullfit_run_dir")}</strong>
            <span>Model Hash Match</span>
            <strong><StatusPill value={Boolean(payload?.production_anchor?.model_hash_match)} /></strong>
            <span>Train End Match</span>
            <strong><StatusPill value={Boolean(payload?.production_anchor?.metrics_train_end_match)} /></strong>
            <span>旧路径数量</span>
            <strong>{field(payload?.production_anchor, "old_path_count")}</strong>
          </div>
        </Panel>
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
        <button className="primary" onClick={refresh} disabled={refreshIsSkip} aria-label={refreshButtonLabel}>
          <RefreshCw size={16} />
          {refreshButtonLabel}
        </button>
        {jobMessage ? <p className="inline-message">{jobMessage}</p> : null}
      </Panel>
      {activeJobId ? (
        <Panel title="刷新进度" meta={jobDetail ? <StatusPill value={jobDetail.status} /> : null}>
          {jobError ? <ErrorState message={jobError} /> : null}
          {jobDetail ? (
            <div className="job-detail">
              <div className="key-list">
                <span>Job ID</span>
                <strong>{jobDetail.job_id}</strong>
                <span>任务</span>
                <strong>{jobDetail.task_name}</strong>
                <span>业务状态</span>
                <strong><StatusPill value={jobDetail.business_status || jobDetail.metadata.business_status || "-"} /></strong>
                <span>Runner</span>
                <strong><StatusPill value={jobDetail.runner_status || jobDetail.metadata.runner_status || "-"} /></strong>
                <span>Artifact</span>
                <strong><StatusPill value={jobDetail.artifact_status || jobDetail.metadata.artifact_status || "-"} /></strong>
                <span>Signal Manifest</span>
                <strong>{text(jobDetail.metadata.signal_refresh_manifest_path)}</strong>
                <span>Panel Latest</span>
                <strong>{text(jobDetail.metadata.panel_latest_date)}</strong>
              </div>
              {Object.entries(jobDetail.evidence_paths || jobDetail.metadata.evidence_paths || {}).some(([, value]) => String(value || "").trim()) ? (
                <DataTable
                  rows={Object.entries(jobDetail.evidence_paths || jobDetail.metadata.evidence_paths || {})
                    .filter(([, value]) => String(value || "").trim())
                    .map(([key, value]) => ({ key, path: String(value) }))}
                  preferredColumns={["key", "path"]}
                />
              ) : null}
              <div className="log-grid">
                <div>
                  <h3>stdout</h3>
                  <pre>{jobDetail.stdout_tail.join("\n") || "-"}</pre>
                </div>
                <div>
                  <h3>stderr</h3>
                  <pre>{jobDetail.stderr_tail.join("\n") || "-"}</pre>
                </div>
              </div>
            </div>
          ) : (
            <LoadingState label="等待作业日志" />
          )}
        </Panel>
      ) : null}
      <details className="panel">
        <summary>历史 Lake Datasets</summary>
        <DataTable
          rows={(payload?.datasets || []).map((row) => ({ ...row }))}
          preferredColumns={["dataset_id", "dataset_kind", "domain", "zone", "status", "start_date", "end_date", "created_at"]}
          emptyText="暂无 dataset"
        />
      </details>
    </div>
  );
}
