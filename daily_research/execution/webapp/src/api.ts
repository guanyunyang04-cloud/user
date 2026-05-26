import type {
  AccountPayload,
  AccountSaveRequest,
  DataRefreshRequest,
  DataSourcesPayload,
  DoctorPayload,
  ExecutionApi,
  JobDetail,
  JobLaunchPayload,
  JobSummary,
  ModelDetailPayload,
  ModelsPayload,
  PaperAccountPayload,
  PaperCashFlowRequest,
  PaperManualAdjustmentRequest,
  PaperPerformancePayload,
  ProviderHealthRequest,
  SchedulerConfigRequest,
  SchedulerPayload,
  StatusPayload,
  TradePlanGenerateRequest,
  TradePlanPayload,
  TrainModelRequest
} from "./types";

type FetchLike = typeof fetch;

async function requestJson<T>(fetcher: FetchLike, url: string, init: RequestInit = {}): Promise<T> {
  const response = await fetcher(url, {
    method: "GET",
    headers: { "Content-Type": "application/json", ...(init.headers || {}) },
    ...init
  });
  if (!response.ok) {
    let message = `${response.status} ${response.statusText}`;
    try {
      const payload = await response.json();
      if (payload?.detail) {
        message = String(payload.detail);
      }
    } catch {
      // Keep the HTTP status message.
    }
    throw new Error(message);
  }
  return response.json() as Promise<T>;
}

function postJson<T>(fetcher: FetchLike, url: string, body: unknown): Promise<T> {
  return requestJson<T>(fetcher, url, {
    method: "POST",
    body: JSON.stringify(body)
  });
}

function patchJson<T>(fetcher: FetchLike, url: string, body: unknown): Promise<T> {
  return requestJson<T>(fetcher, url, {
    method: "PATCH",
    body: JSON.stringify(body)
  });
}

export function createApiClient(fetcher: FetchLike = window.fetch.bind(window)): ExecutionApi {
  return {
    getStatus: (historyLimit = 10) => requestJson<StatusPayload>(fetcher, `/api/status?history_limit=${historyLimit}`),
    getDoctor: () => requestJson<DoctorPayload>(fetcher, "/api/doctor"),
    getModels: () => requestJson<ModelsPayload>(fetcher, "/api/models"),
    getModelDetail: (modelId: string) => requestJson<ModelDetailPayload>(fetcher, `/api/models/${encodeURIComponent(modelId)}`),
    trainModel: (modelId: string, payload: TrainModelRequest) =>
      postJson<JobLaunchPayload>(fetcher, `/api/models/${encodeURIComponent(modelId)}/train`, payload),
    getDataSources: () => requestJson<DataSourcesPayload>(fetcher, "/api/data-sources"),
    refreshDataSources: (payload: DataRefreshRequest) => postJson<JobLaunchPayload>(fetcher, "/api/data-sources/refresh", payload),
    runProviderHealth: (payload: ProviderHealthRequest) =>
      postJson<JobLaunchPayload>(fetcher, "/api/data-sources/provider-health", payload),
    getScheduler: () => requestJson<SchedulerPayload>(fetcher, "/api/data-sources/scheduler"),
    updateScheduler: (payload: SchedulerConfigRequest) => patchJson<SchedulerPayload>(fetcher, "/api/data-sources/scheduler", payload),
    getTradePlan: () => requestJson<TradePlanPayload>(fetcher, "/api/trade-plan"),
    generateTradePlan: (payload: TradePlanGenerateRequest) => postJson<JobLaunchPayload>(fetcher, "/api/trade-plan/generate", payload),
    getAccount: () => requestJson<AccountPayload>(fetcher, "/api/account"),
    saveAccount: (payload: AccountSaveRequest) => postJson<AccountPayload>(fetcher, "/api/account", payload),
    resetAccountExample: () => postJson<AccountPayload>(fetcher, "/api/account/reset-example", {}),
    getPaperAccount: () => requestJson<PaperAccountPayload>(fetcher, "/api/paper-account"),
    recordPaperCashFlow: (payload: PaperCashFlowRequest) => postJson<PaperAccountPayload>(fetcher, "/api/paper-account/cash-flow", payload),
    recordPaperManualAdjustment: (payload: PaperManualAdjustmentRequest) =>
      postJson<PaperAccountPayload>(fetcher, "/api/paper-account/manual-adjustment", payload),
    applyLatestPaperPlan: (payload = {}) => postJson(fetcher, "/api/paper-account/apply-latest-plan", payload),
    getPaperPerformance: (startDate: string, endDate: string) =>
      requestJson<PaperPerformancePayload>(
        fetcher,
        `/api/paper-account/performance?start_date=${encodeURIComponent(startDate)}&end_date=${encodeURIComponent(endDate)}`
      ),
    getJobs: (limit = 30) => requestJson<JobSummary[]>(fetcher, `/api/jobs?limit=${limit}`),
    getJob: (jobId: string, lines = 160) => requestJson<JobDetail>(fetcher, `/api/jobs/${encodeURIComponent(jobId)}?lines=${lines}`),
    streamJob: (jobId, handlers) => {
      if (typeof EventSource === "undefined") {
        const error = new Error("EventSource is not available");
        window.setTimeout(() => handlers.onError?.(error), 0);
        return { close: () => undefined };
      }
      const source = new EventSource(`/api/jobs/${encodeURIComponent(jobId)}/stream`);
      const forward = (event: MessageEvent): void => {
        try {
          handlers.onEvent?.(JSON.parse(event.data));
        } catch (err) {
          handlers.onError?.(err instanceof Error ? err : new Error("Invalid stream event"));
        }
      };
      ["snapshot", "progress", "stdout", "stderr", "status", "done", "error"].forEach((eventName) => {
        source.addEventListener(eventName, forward as EventListener);
      });
      source.onopen = () => handlers.onOpen?.();
      source.onerror = (event) => handlers.onError?.(event);
      return { close: () => source.close() };
    },
    resumeJob: (jobId: string) => postJson<JobLaunchPayload>(fetcher, "/api/resume", { job_id: jobId, background: true }),
    unlockRuntime: (force: boolean) => postJson<{ status: string; detail: string }>(fetcher, "/api/unlock", { force })
  };
}

export const apiClient = createApiClient();
