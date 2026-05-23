import { useEffect, useState } from "react";
import { RotateCcw } from "lucide-react";
import type { ExecutionApi, JobSummary } from "../types";
import { DataTable, ErrorState, LoadingState, PageHeader, Panel, StatusPill } from "../components";

interface JobsPageProps {
  api: ExecutionApi;
  pollMs?: number;
}

export function JobsPage({ api, pollMs = 5000 }: JobsPageProps): JSX.Element {
  const [jobs, setJobs] = useState<JobSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

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

  return (
    <div>
      <PageHeader title="作业" eyebrow="任务历史、状态轮询与恢复入口" actions={<button onClick={load}><RotateCcw size={16} />刷新</button>} />
      {loading ? <LoadingState /> : null}
      {error ? <ErrorState message={error} /> : null}
      <Panel title="最近作业">
        <div className="job-list">
          {jobs.map((job) => (
            <div className="job-row" key={job.job_id}>
              <strong>{job.task_name}</strong>
              <code>{job.job_id}</code>
              <StatusPill value={job.status} />
              <span>{job.created_at || job.started_at || ""}</span>
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
