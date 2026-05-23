export type JsonObject = Record<string, unknown>;

export interface ModelCard {
  id: string;
  role: string;
  name: string;
  status: string;
  usage: string;
  is_live: boolean;
  is_shadow: boolean;
  is_legacy: boolean;
  data_source: string;
  dataset_id: string;
  trained_at: string;
  train_start_date: string;
  train_end_date: string;
  signal_date: string;
  artifact_path: string;
  detail: JsonObject;
}

export interface ModelsPayload {
  status: string;
  models: ModelCard[];
  updated_at?: string;
}

export interface ModelDetailPayload {
  status: string;
  model: ModelCard;
}

export interface TrainModelRequest {
  dataset_mode: "latest" | "dataset_id" | "custom";
  dataset_id?: string;
  start_date?: string;
  end_date?: string;
  job_label?: string;
  force_unlock?: boolean;
  advanced_args?: string;
}

export interface DataSourceRow {
  dataset_id: string;
  dataset_kind: string;
  domain: string;
  zone: string;
  status: string;
  start_date: string;
  end_date: string;
  created_at: string;
}

export interface DataSourcesPayload {
  status: string;
  lake_root: string;
  catalog_status: string;
  datasets: DataSourceRow[];
  data_platform: {
    runs_root: string;
    latest_refresh_run: string;
    provider_plan: string;
    latest_completed_trading_date?: string;
    recommended_domains?: string[];
    default_refresh?: {
      as_of_date: string;
      universe: string;
      domains: string[];
      provider_plan: string;
    };
  };
}

export interface DataRefreshRequest {
  as_of_date?: string;
  start_date?: string;
  universe?: string;
  domains?: string[];
  job_label?: string;
  force_unlock?: boolean;
  advanced_args?: string;
}

export type TableRow = Record<string, string | number | null | undefined>;

export interface TradePlanPayload {
  path?: string;
  exists: boolean;
  status: string;
  summary: JsonObject;
  actions: TableRow[];
  holdings: TableRow[];
  watchlist: TableRow[];
  model_info: JsonObject;
  txt_preview: string[];
  artifact_paths: Record<string, string>;
}

export interface TradePlanGenerateRequest {
  candidate_profile?: string;
  positions_file?: string;
  cash?: string | number | null;
  lot_size?: string | number | null;
  target_weight_top_k?: string | number | null;
  target_weight_min_weight?: string | number | null;
  raw_args_text?: string;
  job_label?: string;
  force_unlock?: boolean;
}

export interface AccountPosition {
  stock: string;
  shares: number | string;
  cost_price: number | string;
}

export interface AccountPayload {
  path: string;
  exists: boolean;
  available_cash: number | null;
  positions: AccountPosition[];
  position_count: number;
  total_shares: number;
  last_modified_at: string;
}

export interface AccountSaveRequest {
  available_cash: number | string | null;
  positions: AccountPosition[];
}

export interface JobSummary {
  job_id: string;
  task_name: string;
  status: string;
  created_at?: string;
  started_at?: string;
  finished_at?: string;
  exit_code?: number;
  summary_note?: string;
  command_argv?: string[];
  [key: string]: unknown;
}

export interface JobDetail {
  job_id: string;
  task_name: string;
  status: string;
  metadata: JobSummary;
  stdout_tail: string[];
  stderr_tail: string[];
  can_resume: boolean;
}

export interface JobLaunchPayload {
  job_id: string;
  task_name?: string;
  status: string;
  command?: string[];
  stdout_log?: string;
  stderr_log?: string;
  [key: string]: unknown;
}

export interface StatusPayload {
  runtime_root: string;
  current_job?: JsonObject;
  recent_jobs?: JobSummary[];
  active_manifest?: JsonObject;
  current_positions?: JsonObject;
  latest_trade_plan?: TradePlanPayload;
  warnings?: string[];
  lock?: JsonObject;
  updated_at?: string;
  [key: string]: unknown;
}

export interface DoctorPayload {
  status: string;
  checked_at: string;
  checks: Array<{ name: string; ok: boolean; detail: string }>;
}

export interface ExecutionApi {
  getStatus(historyLimit?: number): Promise<StatusPayload>;
  getDoctor(): Promise<DoctorPayload>;
  getModels(): Promise<ModelsPayload>;
  getModelDetail(modelId: string): Promise<ModelDetailPayload>;
  trainModel(modelId: string, payload: TrainModelRequest): Promise<JobLaunchPayload>;
  getDataSources(): Promise<DataSourcesPayload>;
  refreshDataSources(payload: DataRefreshRequest): Promise<JobLaunchPayload>;
  getTradePlan(): Promise<TradePlanPayload>;
  generateTradePlan(payload: TradePlanGenerateRequest): Promise<JobLaunchPayload>;
  getAccount(): Promise<AccountPayload>;
  saveAccount(payload: AccountSaveRequest): Promise<AccountPayload>;
  resetAccountExample(): Promise<AccountPayload>;
  getJobs(limit?: number): Promise<JobSummary[]>;
  getJob(jobId: string, lines?: number): Promise<JobDetail>;
  resumeJob(jobId: string): Promise<JobLaunchPayload>;
  unlockRuntime(force: boolean): Promise<{ status: string; detail: string }>;
}
