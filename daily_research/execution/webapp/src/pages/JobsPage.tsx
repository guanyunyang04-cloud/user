import { useEffect, useState } from "react";
import { RotateCcw } from "lucide-react";
import type { ExecutionApi, JobDetail, JobSummary } from "../types";
import { DataTable, ErrorState, LoadingState, PageHeader, Panel, StatusPill } from "../components";
import { text } from "../format";

interface JobsPageProps {
  api: ExecutionApi;
  pollMs?: number;
  selectedJobId?: string;
}

function objectRows(payload: Record<string, string> | undefined): Array<{ key: string; path: string }> {
  return Object.entries(payload || {})
    .filter(([, value]) => String(value || "").trim())
    .map(([key, value]) => ({ key, path: String(value) }));
}

export function JobsPage({ api, pollMs = 5000, selectedJobId = "" }: JobsPageProps): JSX.Element {
  const [jobs, setJobs] = useState<JobSummary[]>([]);
  const [selectedId, setSelectedId] = useState(selectedJobId);
  const [jobDetail, setJobDetail] = useState<JobDetail | null>(null);
  const [loading, setLoading] = useState(true);
  const [detailLoading, setDetailLoading] = useState(false);
  const [error, setError] = useState("");
  const [detailError, setDetailError] = useState("");

  const load = (): Promise<void> => {
    return api
      .getJobs(40)
      .then((next) => {
        setJobs(next);
        setError("");
      })
      .catch((err: Error) => setError(err.message))
      .finally(() => setLoading(false));
  };

  const loadDetail = (jobId: string, updateUrl = true): Promise<void> => {
    if (!jobId) {
      setSelectedId("");
      setJobDetail(null);
      setDetailError("");
      return Promise.resolve();
    }
    setSelectedId(jobId);
    if (updateUrl && window.location.pathname !== `/jobs/${encodeURIComponent(jobId)}`) {
      window.history.pushState(null, "", `/jobs/${encodeURIComponent(jobId)}`);
    }
    setDetailLoading(true);
    return api
      .getJob(jobId, 200)
      .then((next) => {
        setJobDetail(next);
        setDetailError("");
      })
      .catch((err: Error) => setDetailError(err.message))
      .finally(() => setDetailLoading(false));
  };

  useEffect(() => {
    let disposed = false;
    let timer: number | undefined;
    const run = (): void => {
      load().finally(() => {
        if (!disposed) {
          timer = window.setTimeout(run, pollMs);
        }
      });
    };
    run();
    return () => {
      disposed = true;
      if (timer !== undefined) {
        window.clearTimeout(timer);
      }
    };
  }, [api, pollMs]);

  useEffect(() => {
    setSelectedId(selectedJobId);
  }, [selectedJobId]);

  useEffect(() => {
    let disposed = false;
    let timer: number | undefined;
    const run = (): void => {
      if (!selectedId) {
        setJobDetail(null);
        setDetailError("");
        return;
      }
      api
        .getJob(selectedId, 200)
        .then((next) => {
          if (!disposed) {
            setJobDetail(next);
            setDetailError("");
          }
        })
        .catch((err: Error) => {
          if (!disposed) {
            setDetailError(err.message);
          }
        })
        .finally(() => {
          if (!disposed) {
            setDetailLoading(false);
            timer = window.setTimeout(run, pollMs);
          }
        });
    };
    if (selectedId) {
      setDetailLoading(true);
    }
    run();
    return () => {
      disposed = true;
      if (timer !== undefined) {
        window.clearTimeout(timer);
      }
    };
  }, [api, selectedId, pollMs]);

  return (
    <div>
      <PageHeader title="作业" eyebrow="任务历史、状态轮询与恢复入口" actions={<button onClick={load}><RotateCcw size={16} />刷新</button>} />
      {loading ? <LoadingState /> : null}
      {error ? <ErrorState message={error} /> : null}
      {selectedId ? (
        <Panel title="作业详情">
          {detailLoading ? <LoadingState label="加载作业详情" /> : null}
          {detailError ? <ErrorState message={detailError} /> : null}
          {jobDetail ? (
            <div className="job-detail">
              <div className="key-list">
                <span>Job ID</span>
                <strong>{jobDetail.job_id}</strong>
                <span>任务</span>
                <strong>{jobDetail.task_name}</strong>
                <span>状态</span>
                <strong><StatusPill value={jobDetail.status} /></strong>
                <span>业务状态</span>
                <strong><StatusPill value={jobDetail.business_status || jobDetail.metadata.business_status || "-"} /></strong>
                <span>Runner</span>
                <strong><StatusPill value={jobDetail.runner_status || jobDetail.metadata.runner_status || "-"} /></strong>
                <span>Artifact</span>
                <strong><StatusPill value={jobDetail.artifact_status || jobDetail.metadata.artifact_status || "-"} /></strong>
                <span>命令</span>
                <strong>{(jobDetail.metadata.command_argv || []).join(" ") || "-"}</strong>
              </div>
              {objectRows(jobDetail.evidence_paths || jobDetail.metadata.evidence_paths).length ? (
                <div>
                  <h3>Evidence</h3>
                  <DataTable
                    rows={objectRows(jobDetail.evidence_paths || jobDetail.metadata.evidence_paths)}
                    preferredColumns={["key", "path"]}
                  />
                </div>
              ) : null}
              {(jobDetail.runner_warnings || jobDetail.metadata.runner_warnings || []).length ? (
                <div>
                  <h3>Runner Warnings</h3>
                  <ul className="compact-list">
                    {(jobDetail.runner_warnings || jobDetail.metadata.runner_warnings || []).map((warning) => (
                      <li key={warning}>{warning}</li>
                    ))}
                  </ul>
                </div>
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
              {jobDetail.can_resume ? (
                <button onClick={() => api.resumeJob(jobDetail.job_id)}>重跑/恢复 {jobDetail.job_id}</button>
              ) : null}
            </div>
          ) : null}
        </Panel>
      ) : null}
      <Panel title="最近作业">
        <div className="job-list">
          {jobs.map((job) => (
            <div className="job-row" key={job.job_id}>
              <strong>{job.task_name}</strong>
              <code>{job.job_id}</code>
              <StatusPill value={job.status} />
              <StatusPill value={job.business_status || "-"} />
              <StatusPill value={job.runner_status || "-"} />
              <span>{job.created_at || job.started_at || ""}</span>
              <button onClick={() => loadDetail(job.job_id)}>查看</button>
              {["failed", "blocked", "succeeded"].includes(job.status) ? (
                <button onClick={() => api.resumeJob(job.job_id)}>重跑/恢复 {job.job_id}</button>
              ) : null}
            </div>
          ))}
        </div>
        <DataTable
          rows={jobs.map((job) => ({
            job_id: job.job_id,
            task_name: job.task_name,
            status: job.status,
            business_status: text(job.business_status),
            runner_status: text(job.runner_status),
            artifact_status: text(job.artifact_status),
            started_at: job.started_at,
            completed_at: job.completed_at || job.finished_at,
            exit_code: job.exit_code
          }))}
          preferredColumns={["job_id", "task_name", "status", "business_status", "runner_status", "artifact_status", "started_at", "completed_at", "exit_code"]}
          emptyText="暂无作业"
        />
      </Panel>
    </div>
  );
}
