import { useEffect, useState } from "react";
import { RotateCcw } from "lucide-react";
import type { ExecutionApi, JobDetail, JobSummary } from "../types";
import { DataTable, ErrorState, LoadingState, PageHeader, Panel, StatusPill } from "../components";

interface JobsPageProps {
  api: ExecutionApi;
  pollMs?: number;
  selectedJobId?: string;
}

export function JobsPage({ api, pollMs = 5000, selectedJobId = "" }: JobsPageProps): JSX.Element {
  const [jobs, setJobs] = useState<JobSummary[]>([]);
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

  const loadDetail = (jobId: string): Promise<void> => {
    if (!jobId) {
      setJobDetail(null);
      setDetailError("");
      return Promise.resolve();
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
    let disposed = false;
    if (!selectedJobId) {
      setJobDetail(null);
      setDetailError("");
      return () => {
        disposed = true;
      };
    }
    setDetailLoading(true);
    api
      .getJob(selectedJobId, 200)
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
        }
      });
    return () => {
      disposed = true;
    };
  }, [api, selectedJobId]);

  return (
    <div>
      <PageHeader title="作业" eyebrow="任务历史、状态轮询与恢复入口" actions={<button onClick={load}><RotateCcw size={16} />刷新</button>} />
      {loading ? <LoadingState /> : null}
      {error ? <ErrorState message={error} /> : null}
      {selectedJobId ? (
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
                <span>命令</span>
                <strong>{(jobDetail.metadata.command_argv || []).join(" ") || "-"}</strong>
              </div>
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
                <button onClick={() => api.resumeJob(jobDetail.job_id)}>重跑/恢复</button>
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
              <span>{job.created_at || job.started_at || ""}</span>
              <button onClick={() => loadDetail(job.job_id)}>查看</button>
              {["failed", "blocked", "succeeded"].includes(job.status) ? (
                <button onClick={() => api.resumeJob(job.job_id)}>重跑/恢复</button>
              ) : null}
            </div>
          ))}
        </div>
        <DataTable
          rows={jobs.map((job) => ({
            job_id: job.job_id,
            task_name: job.task_name,
            status: job.status,
            started_at: job.started_at,
            finished_at: job.finished_at,
            exit_code: job.exit_code
          }))}
          preferredColumns={["job_id", "task_name", "status", "started_at", "finished_at", "exit_code"]}
          emptyText="暂无作业"
        />
      </Panel>
    </div>
  );
}
