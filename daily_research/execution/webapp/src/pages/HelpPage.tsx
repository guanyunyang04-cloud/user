import { useEffect, useMemo, useState } from "react";
import { RefreshCw } from "lucide-react";
import type {
  DataSourcesPayload,
  ExecutionApi,
  JobSummary,
  PaperAccountPayload,
  StatusPayload,
  TableRow,
  TradePlanPayload
} from "../types";
import { DataTable, ErrorState, LoadingState, PageHeader, Panel, Stat, StatusPill } from "../components";
import { text } from "../format";

interface HelpPageProps {
  api: ExecutionApi;
}

interface HelpPayloads {
  status?: StatusPayload | null;
  dataSources?: DataSourcesPayload | null;
  tradePlan?: TradePlanPayload | null;
  paperAccount?: PaperAccountPayload | null;
  jobs?: JobSummary[];
}

interface GuideState {
  nextStep: string;
  nextStepDetail: string;
  checklistRows: TableRow[];
  warningRows: TableRow[];
}

function hasRuntimeLock(status?: StatusPayload | null): boolean {
  return Boolean(status?.lock && Object.keys(status.lock).length);
}

function hasRunningJob(status?: StatusPayload | null, jobs: JobSummary[] = []): boolean {
  const currentStatus = String(status?.current_job?.status || "").toLowerCase();
  if (["queued", "running"].includes(currentStatus)) {
    return true;
  }
  return jobs.some((job) => ["queued", "running"].includes(String(job.status || "").toLowerCase()));
}

function hasMissingExecutionOpen(paperAccount?: PaperAccountPayload | null): boolean {
  return (paperAccount?.pending_orders || []).some((order) => String(order.reason || "").includes("missing_execution_open"));
}

export function deriveGuideState({
  status,
  dataSources,
  tradePlan,
  paperAccount,
  jobs = []
}: HelpPayloads): GuideState {
  const runtimeBusy = hasRuntimeLock(status) || hasRunningJob(status, jobs);
  const dataNeedsRefresh = dataSources?.next_refresh_action !== "skip";
  const signalNeedsRefresh = dataSources?.next_signal_action === "refresh";
  const tradePlanReady = tradePlan?.status === "ok" && tradePlan.exists !== false;
  const pendingPaperOrders = Number(paperAccount?.pending_order_count || 0) > 0;
  const missingExecutionOpen = hasMissingExecutionOpen(paperAccount);

  let nextStep = "可生成交易计划";
  let nextStepDetail = "数据与信号已最新";
  if (runtimeBusy) {
    nextStep = "先查看运行中作业";
    nextStepDetail = "当前存在运行中 job 或 runtime lock，先到作业/系统页确认状态。";
  } else if (dataNeedsRefresh) {
    nextStep = "先刷新数据";
    nextStepDetail = "当前 active dataset 还没有确认覆盖最新完成交易日。";
  } else if (signalNeedsRefresh) {
    nextStep = "先刷新信号面板";
    nextStepDetail = "数据已最新，但 production signal panel 还没覆盖最新完成交易日。";
  } else if (!tradePlanReady) {
    nextStep = "生成交易计划";
    nextStepDetail = "数据和信号已经就绪，但还没有可展示的最新交易计划。";
  } else if (missingExecutionOpen) {
    nextStep = "等待执行日开盘价入湖";
    nextStepDetail = "模拟订单已注册，但执行日 open 价格缺失，所以不能伪造成交。";
  } else if (pendingPaperOrders) {
    nextStep = "模拟账户过账";
    nextStepDetail = "交易计划订单已注册，等待可用价格后可在账户页模拟过账。";
  }

  const checklistRows: TableRow[] = [
    {
      step: "检查数据状态",
      status: dataNeedsRefresh ? "需要刷新" : "ok",
      action: dataNeedsRefresh ? "去数据页补齐到最新交易日" : "无需操作",
      detail: text(dataSources?.dataset_sync_status || dataSources?.current_dataset_status)
    },
    {
      step: "刷新数据/信号",
      status: signalNeedsRefresh ? "需要刷新信号" : dataNeedsRefresh ? "需要刷新数据" : "ok",
      action: signalNeedsRefresh ? "去数据页刷新 signal panel" : dataNeedsRefresh ? "去数据页刷新" : "无需重复刷新",
      detail: text(dataSources?.signal_panel_status)
    },
    {
      step: "生成交易计划",
      status: tradePlanReady ? "ok" : "missing",
      action: tradePlanReady ? "检查结构化动作与 TXT 原文" : "去交易计划页生成",
      detail: `${tradePlan?.actions?.length || 0} 个动作`
    },
    {
      step: "模拟账户过账",
      status: missingExecutionOpen ? "pending" : pendingPaperOrders ? "pending" : "ok",
      action: missingExecutionOpen ? "等 open 价格可用后过账" : pendingPaperOrders ? "去账户页模拟过账" : "无待处理订单",
      detail: `${paperAccount?.pending_order_count || 0} 个未成交订单`
    },
    {
      step: "查看收益",
      status: paperAccount?.latest_equity ? "ok" : "waiting",
      action: "去账户页查询区间收益",
      detail: text(paperAccount?.latest_equity?.as_of_date)
    }
  ];

  const warningRows: TableRow[] = [];
  if (runtimeBusy) {
    warningRows.push({ item: "运行中作业", meaning: "当前不建议提交新任务", where: "作业 / 系统" });
  }
  if (missingExecutionOpen) {
    warningRows.push({ item: "missing_execution_open", meaning: "执行日开盘价缺失，模拟订单保持 pending", where: "账户" });
  }
  if (signalNeedsRefresh) {
    warningRows.push({ item: "signal stale", meaning: "信号面板未覆盖最新完成交易日", where: "数据" });
  }
  return { nextStep, nextStepDetail, checklistRows, warningRows };
}

const PAGE_ROWS: TableRow[] = [
  { page: "总览", use: "看 active manifest、交易计划、模拟账户和最近作业摘要", safe_action: "只读检查" },
  { page: "模型", use: "确认当前 live/production/research/legacy 模型状态", safe_action: "显式训练才会提交任务" },
  { page: "数据", use: "检查当前数据集、provider 健康、signal panel、盘后自动更新", safe_action: "刷新数据或信号" },
  { page: "交易计划", use: "生成并查看结构化动作、watchlist、市场状态和 TXT 原文", safe_action: "注册模拟订单" },
  { page: "模拟账户", use: "查看现金、持仓、pending orders、成交流水和区间收益", safe_action: "充值提现、手动修正、模拟过账" },
  { page: "作业", use: "查看任务历史、runner/business/artifact 状态和日志", safe_action: "查看详情或恢复失败任务" },
  { page: "系统", use: "查看 doctor、runtime、锁、scheduler 和关键证据路径", safe_action: "必要时清理锁" }
];

const STATUS_ROWS: TableRow[] = [
  { status: "synced", meaning: "active dataset 与最新 policy input dataset 一致", action: "通常无需刷新数据" },
  { status: "stale", meaning: "数据集或信号面板落后于最新完成交易日", action: "去数据页刷新" },
  { status: "pending", meaning: "订单已注册但未模拟成交", action: "等待价格或手动检查账户页" },
  { status: "missing_execution_open", meaning: "执行日 open 价格缺失", action: "刷新数据后再过账，不伪造成交" },
  { status: "runner warning", meaning: "业务产物可能成功，但 runner 层有输出或日志警告", action: "看作业详情的业务状态和证据路径" },
  { status: "blocked", meaning: "任务被锁、缺依赖或业务前置条件阻止", action: "先处理 blocker" },
  { status: "succeeded_with_warning", meaning: "业务完成但有非阻断警告", action: "复核 warnings 和 artifact paths" }
];

const FAQ_ROWS: TableRow[] = [
  { question: "为什么重复点刷新不重新跑全量？", answer: "当前 dataset 已覆盖最新完成交易日时会跳过真实刷新；若 signal stale，只刷新信号面板。" },
  { question: "为什么交易计划没有动作？", answer: "看交易计划页的诊断：目标仓位、候选行、score context 和市场过滤都会解释原因。" },
  { question: "为什么模拟订单没成交？", answer: "默认按执行日开盘价成交；open 缺失、停牌、涨跌停或现金不足时订单保持 pending/blocked。" },
  { question: "作业 warning 是失败吗？", answer: "不一定。作业页要同时看业务状态、runner 状态、artifact 状态和 evidence paths。" }
];

export function HelpPage({ api }: HelpPageProps): JSX.Element {
  const [status, setStatus] = useState<StatusPayload | null>(null);
  const [dataSources, setDataSources] = useState<DataSourcesPayload | null>(null);
  const [tradePlan, setTradePlan] = useState<TradePlanPayload | null>(null);
  const [paperAccount, setPaperAccount] = useState<PaperAccountPayload | null>(null);
  const [jobs, setJobs] = useState<JobSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  const load = (): void => {
    setLoading(true);
    Promise.all([api.getStatus(10), api.getDataSources(), api.getTradePlan(), api.getPaperAccount(), api.getJobs(10)])
      .then(([nextStatus, nextDataSources, nextTradePlan, nextPaperAccount, nextJobs]) => {
        setStatus(nextStatus);
        setDataSources(nextDataSources);
        setTradePlan(nextTradePlan);
        setPaperAccount(nextPaperAccount);
        setJobs(nextJobs);
        setError("");
      })
      .catch((err: Error) => setError(err.message))
      .finally(() => setLoading(false));
  };

  useEffect(() => {
    load();
  }, []);

  const guide = useMemo(
    () => deriveGuideState({ status, dataSources, tradePlan, paperAccount, jobs }),
    [status, dataSources, tradePlan, paperAccount, jobs]
  );

  return (
    <div>
      <PageHeader
        title="帮助"
        eyebrow="日常盘后 runbook、状态解释与安全边界"
        actions={<button onClick={load}><RefreshCw size={16} />刷新状态</button>}
      />
      {loading ? <LoadingState /> : null}
      {error ? <ErrorState message={error} /> : null}

      <div className="stat-grid">
        <Stat label="下一步" value="查看流程" />
        <Stat label="数据" value={<StatusPill value={dataSources?.dataset_sync_status || dataSources?.next_refresh_action || "unknown"} />} />
        <Stat label="信号面板" value={<StatusPill value={dataSources?.signal_panel_status || "unknown"} />} />
        <Stat label="交易计划" value={<StatusPill value={tradePlan?.status || "missing"} />} />
        <Stat label="未成交订单" value={paperAccount?.pending_order_count || 0} />
        <Stat label="运行状态" value={<StatusPill value={hasRuntimeLock(status) || hasRunningJob(status, jobs) ? "running" : "ok"} />} />
      </div>

      <Panel title="今日盘后流程">
        <div className="guide-next-step">
          <strong>{guide.nextStep}</strong>
          <span>{guide.nextStepDetail}</span>
        </div>
        <DataTable
          rows={guide.checklistRows}
          preferredColumns={["step", "status", "action", "detail"]}
          emptyText="暂无流程状态"
        />
      </Panel>

      {guide.warningRows.length ? (
        <Panel title="当前提示">
          <DataTable rows={guide.warningRows} preferredColumns={["item", "meaning", "where"]} />
        </Panel>
      ) : null}

      <Panel title="页面说明">
        <DataTable rows={PAGE_ROWS} preferredColumns={["page", "use", "safe_action"]} />
      </Panel>

      <Panel title="状态词典">
        <DataTable rows={STATUS_ROWS} preferredColumns={["status", "meaning", "action"]} />
      </Panel>

      <Panel title="安全边界">
        <div className="guide-boundaries">
          <span>不真实下单</span>
          <span>不自动重训</span>
          <span>不 promotion</span>
          <span>不 activation</span>
          <span>只读状态不会触发任务</span>
        </div>
      </Panel>

      <Panel title="常见问题">
        <DataTable rows={FAQ_ROWS} preferredColumns={["question", "answer"]} />
      </Panel>
    </div>
  );
}
