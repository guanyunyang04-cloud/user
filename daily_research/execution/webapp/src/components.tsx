import type { ReactNode } from "react";
import { AlertCircle, CheckCircle2, Loader2 } from "lucide-react";
import type { JobProgress, TableRow } from "./types";
import { tableColumns, text } from "./format";

interface PageHeaderProps {
  title: string;
  eyebrow?: string;
  actions?: ReactNode;
}

export function PageHeader({ title, eyebrow, actions }: PageHeaderProps): JSX.Element {
  return (
    <header className="page-header">
      <div>
        {eyebrow ? <p className="eyebrow">{eyebrow}</p> : null}
        <h1>{title}</h1>
      </div>
      <div className="header-actions">{actions}</div>
    </header>
  );
}

export function Panel({ title, children, meta }: { title?: string; children: ReactNode; meta?: ReactNode }): JSX.Element {
  return (
    <section className="panel">
      {title ? (
        <div className="panel-title">
          <h2>{title}</h2>
          {meta ? <div className="panel-meta">{meta}</div> : null}
        </div>
      ) : null}
      {children}
    </section>
  );
}

export function Stat({ label, value, tone = "neutral" }: { label: string; value: ReactNode; tone?: string }): JSX.Element {
  return (
    <div className={`stat stat-${tone}`}>
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

export function StatusPill({ value }: { value: string | boolean | undefined }): JSX.Element {
  const label = typeof value === "boolean" ? (value ? "ok" : "fail") : text(value, "unknown");
  const normalized = label.toLowerCase();
  const tone = ["ok", "active", "succeeded", "safe", "fresh", "synced"].includes(normalized)
    ? "good"
    : ["failed", "danger", "blocked", "fail"].includes(normalized)
      ? "bad"
      : ["running", "queued", "manual", "shadow", "caution", "warning", "stale"].includes(normalized)
        ? "wait"
        : "neutral";
  return <span className={`pill pill-${tone}`}>{label}</span>;
}

export function LoadingState({ label = "加载中" }: { label?: string }): JSX.Element {
  return (
    <div className="state-line">
      <Loader2 size={16} className="spin" />
      <span>{label}</span>
    </div>
  );
}

export function ErrorState({ message }: { message: string }): JSX.Element {
  return (
    <div className="state-line state-error">
      <AlertCircle size={16} />
      <span>{message}</span>
    </div>
  );
}

export function EmptyState({ title, detail }: { title: string; detail?: string }): JSX.Element {
  return (
    <div className="empty-state">
      <CheckCircle2 size={18} />
      <strong>{title}</strong>
      {detail ? <span>{detail}</span> : null}
    </div>
  );
}

export function DataTable({
  rows,
  preferredColumns = [],
  emptyText = "暂无数据"
}: {
  rows: TableRow[];
  preferredColumns?: string[];
  emptyText?: string;
}): JSX.Element {
  if (!rows.length) {
    return <EmptyState title={emptyText} />;
  }
  const columns = tableColumns(rows, preferredColumns);
  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            {columns.map((column) => (
              <th key={column}>{column}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, rowIndex) => (
            <tr key={`${text(row.stock || row.code || row.dataset_id, "row")}-${rowIndex}`}>
              {columns.map((column) => (
                <td key={column}>{text(row[column])}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function ProgressBar({ progress }: { progress?: JobProgress | null }): JSX.Element {
  const total = Number(progress?.total_steps || 0);
  const completed = Number(progress?.completed_steps || 0);
  const rawPercent = progress?.percent !== undefined ? Number(progress.percent) : total > 0 ? (completed / total) * 100 : 0;
  const percentValue = Number.isFinite(rawPercent) ? Math.max(0, Math.min(100, rawPercent)) : 0;
  const determinate = total > 0 || String(progress?.mode || "") === "determinate";
  return (
    <div className="progress-block">
      <div className={`progress-bar ${determinate ? "progress-determinate" : "progress-indeterminate"}`}>
        <span style={determinate ? { width: `${percentValue}%` } : undefined} />
      </div>
      <div className="progress-meta">
        <strong>{determinate ? `${Math.round(percentValue)}%` : "运行中"}</strong>
        <span>{text(progress?.stage || progress?.current_item || "正在运行")}</span>
        {total > 0 ? <span>{completed}/{total}</span> : null}
      </div>
    </div>
  );
}

export function LogDisclosure({ stdout, stderr }: { stdout: string[]; stderr: string[] }): JSX.Element {
  return (
    <details className="log-disclosure">
      <summary>实时日志 / 调试信息</summary>
      <div className="log-grid">
        <div>
          <h3>stdout</h3>
          <pre>{stdout.join("\n") || "-"}</pre>
        </div>
        <div>
          <h3>stderr</h3>
          <pre>{stderr.join("\n") || "-"}</pre>
        </div>
      </div>
    </details>
  );
}

export function ResizableTablePanel({
  rows,
  preferredColumns = [],
  emptyText = "暂无数据"
}: {
  rows: TableRow[];
  preferredColumns?: string[];
  emptyText?: string;
}): JSX.Element {
  return (
    <div className="resizable-table">
      <DataTable rows={rows} preferredColumns={preferredColumns} emptyText={emptyText} />
    </div>
  );
}

export function Field({ label, children }: { label: string; children: ReactNode }): JSX.Element {
  return (
    <label className="field">
      <span>{label}</span>
      {children}
    </label>
  );
}
