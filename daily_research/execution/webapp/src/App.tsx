import { useMemo, useState } from "react";
import {
  Activity,
  BriefcaseBusiness,
  Database,
  LayoutDashboard,
  ListChecks,
  MonitorCog,
  PieChart,
  WalletCards
} from "lucide-react";
import type { ExecutionApi } from "./types";
import { apiClient } from "./api";
import { OverviewPage } from "./pages/OverviewPage";
import { ModelsPage } from "./pages/ModelsPage";
import { DataPage } from "./pages/DataPage";
import { TradePlanPage } from "./pages/TradePlanPage";
import { AccountPage } from "./pages/AccountPage";
import { JobsPage } from "./pages/JobsPage";
import { SystemPage } from "./pages/SystemPage";

type PageKey = "overview" | "models" | "data" | "trade-plan" | "account" | "jobs" | "system";

const NAV_ITEMS: Array<{ key: PageKey; label: string; icon: typeof LayoutDashboard; path: string }> = [
  { key: "overview", label: "总览", icon: LayoutDashboard, path: "/" },
  { key: "models", label: "模型", icon: Activity, path: "/models" },
  { key: "data", label: "数据", icon: Database, path: "/data" },
  { key: "trade-plan", label: "交易计划", icon: PieChart, path: "/trade-plan" },
  { key: "account", label: "账户", icon: WalletCards, path: "/account" },
  { key: "jobs", label: "作业", icon: ListChecks, path: "/jobs" },
  { key: "system", label: "系统", icon: MonitorCog, path: "/system" }
];

function pageFromPath(pathname: string): PageKey {
  const normalized = pathname.replace(/\/+$/, "") || "/";
  if (normalized === "/models") return "models";
  if (normalized === "/data") return "data";
  if (normalized === "/trade-plan" || normalized === "/artifacts/trade-plan") return "trade-plan";
  if (normalized === "/account") return "account";
  if (normalized === "/jobs" || normalized.startsWith("/jobs/")) return "jobs";
  if (normalized === "/system" || normalized === "/doctor" || normalized === "/settings/runtime") return "system";
  return "overview";
}

function selectedJobIdFromPath(pathname: string): string {
  const normalized = pathname.replace(/\/+$/, "");
  if (!normalized.startsWith("/jobs/")) {
    return "";
  }
  return decodeURIComponent(normalized.slice("/jobs/".length));
}

interface AppProps {
  api?: ExecutionApi;
}

export function App({ api = apiClient }: AppProps): JSX.Element {
  const [page, setPage] = useState<PageKey>(() => pageFromPath(window.location.pathname));
  const selectedJobId = page === "jobs" ? selectedJobIdFromPath(window.location.pathname) : "";
  const activeItem = useMemo(() => NAV_ITEMS.find((item) => item.key === page) || NAV_ITEMS[0], [page]);

  function navigate(next: PageKey): void {
    const item = NAV_ITEMS.find((candidate) => candidate.key === next) || NAV_ITEMS[0];
    window.history.pushState(null, "", item.path);
    setPage(next);
  }

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <BriefcaseBusiness size={22} />
          <div>
            <strong>Daily Research</strong>
            <span>执行控制台</span>
          </div>
        </div>
        <nav aria-label="主导航">
          {NAV_ITEMS.map((item) => {
            const Icon = item.icon;
            return (
              <button key={item.key} className={item.key === page ? "nav-active" : ""} onClick={() => navigate(item.key)}>
                <Icon size={17} />
                <span>{item.label}</span>
              </button>
            );
          })}
        </nav>
        <div className="sidebar-foot">
          <span>当前</span>
          <strong>{activeItem.label}</strong>
        </div>
      </aside>
      <main className="workspace">
        {page === "overview" ? <OverviewPage api={api} /> : null}
        {page === "models" ? <ModelsPage api={api} /> : null}
        {page === "data" ? <DataPage api={api} /> : null}
        {page === "trade-plan" ? <TradePlanPage api={api} /> : null}
        {page === "account" ? <AccountPage api={api} /> : null}
        {page === "jobs" ? <JobsPage api={api} selectedJobId={selectedJobId} /> : null}
        {page === "system" ? <SystemPage api={api} /> : null}
      </main>
    </div>
  );
}
