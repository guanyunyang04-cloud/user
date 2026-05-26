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

export interface DomainMatrixRow {
  provider: string;
  domain: string;
  requirement?: string;
  supported?: boolean;
  requires_token?: boolean;
  formal_refresh?: boolean;
  notes?: string;
  [key: string]: unknown;
}

export interface ProviderHealthRequest {
  as_of_date?: string;
  domains?: string[];
  provider_plan?: string;
}

export interface ProviderHealthPayload {
  status: string;
  provider_plan?: string;
  checked_at?: string;
  as_of_date?: string;
  summary?: JsonObject;
  domain_matrix?: DomainMatrixRow[];
  providers?: JsonObject[];
  [key: string]: unknown;
}

export interface JobProgress {
  mode?: "determinate" | "indeterminate" | string;
  stage?: string;
  completed_steps?: number;
  total_steps?: number;
  percent?: number;
  current_item?: string;
  current_provider?: string;
  current_domain?: string;
  updated_at?: string;
  status?: string;
  [key: string]: unknown;
}

export interface DailyRunVerdict {
  schema_version?: number;
  run_date?: string;
  target_trading_date?: string;
  status?: "completed" | "blocked" | "missing" | string;
  blocker_code?: string;
  stage_results?: JsonObject;
  dataset_id?: string;
  signal_panel_date?: string;
  trade_plan_run_dir?: string;
  paper_reconcile_status?: string;
  evidence_paths?: Record<string, string>;
  created_at?: string;
  [key: string]: unknown;
}

export interface DailyRunStatusPayload {
  status: string;
  latest_run_date?: string;
  latest_verdict?: DailyRunVerdict;
  daily_runs_root?: string;
}

export interface DataReadinessPayload {
  status: string;
  blocker_code?: string;
  candidate_date?: string;
  provider_ready_date?: string;
  provider_plan?: string;
  provider?: string;
  row_count?: number;
  coverage_ratio?: number;
  provider_error?: string;
  providers?: JsonObject[];
  [key: string]: unknown;
}

export interface DataSourcesPayload {
  status: string;
  lake_root: string;
  catalog_status: string;
  datasets: DataSourceRow[];
  formal_provider_plan?: string;
  domain_matrix?: DomainMatrixRow[];
  provider_health?: ProviderHealthPayload | JsonObject;
  current_dataset_health?: JsonObject;
  signal_panel_health?: JsonObject;
  active_dataset_id?: string;
  active_dataset_end_date?: string;
  latest_policy_input_dataset_id?: string;
  latest_policy_input_dataset_end_date?: string;
  dataset_sync_status?: string;
  current_dataset_status?: string;
  is_current_dataset_latest?: boolean;
  is_current_dataset_complete?: boolean;
  next_refresh_action?: "skip" | "refresh" | string;
  signal_panel_status?: string;
  signal_panel_latest_date?: string;
  signal_panel_target_latest_date?: string;
  signal_panel_score_latest_date?: string;
  next_signal_action?: "skip" | "refresh" | string;
  production_anchor_status?: string;
  production_anchor?: JsonObject;
  signal_panels?: JsonObject;
  refresh_explanation?: string;
  data_platform: {
    runs_root: string;
    latest_refresh_run: string;
    latest_refresh_manifest_path?: string;
    latest_refresh_manifest_status?: string;
    latest_refresh_registered_dataset_id?: string;
    latest_refresh_blockers?: string[];
    provider_plan: string;
    latest_completed_trading_date?: string;
    recommended_domains?: string[];
    default_refresh?: {
      as_of_date: string;
      universe: string;
      domains: string[];
      required_domains?: string[];
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
  diagnostics?: JsonObject;
  paper_trading?: JsonObject;
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
  status?: string;
  path: string;
  exists: boolean;
  source?: string;
  db_path?: string;
  available_cash: number | null;
  positions: AccountPosition[];
  positions_by_stock?: JsonObject;
  position_count: number;
  total_shares: number;
  last_modified_at: string;
  pending_order_count?: number;
  filled_order_count?: number;
  blocked_order_count?: number;
  pending_orders?: TableRow[];
  recent_fills?: TableRow[];
  recent_cash_flows?: TableRow[];
  latest_equity?: JsonObject;
}

export interface AccountSaveRequest {
  available_cash: number | string | null;
  positions: AccountPosition[];
}

export type PaperAccountPayload = AccountPayload;

export interface PaperCashFlowRequest {
  flow_type: string;
  amount: number | string;
  reason?: string;
}

export interface PaperManualAdjustmentRequest {
  adjustment_type: string;
  stock?: string;
  shares?: number | string | null;
  cost_price?: number | string | null;
  amount?: number | string | null;
  reason?: string;
}

export interface PaperPerformancePayload {
  status: string;
  start_date?: string;
  end_date?: string;
  total_return?: number;
  max_drawdown?: number;
  net_cash_flow?: number;
  points?: TableRow[];
  [key: string]: unknown;
}

export interface JobSummary {
  job_id: string;
  task_name: string;
  status: string;
  business_status?: string;
  runner_status?: string;
  artifact_status?: string;
  artifact_paths?: Record<string, string>;
  evidence_paths?: Record<string, string>;
  runner_warnings?: string[];
  created_at?: string;
  started_at?: string;
  completed_at?: string;
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
  business_status?: string;
  runner_status?: string;
  artifact_status?: string;
  artifact_paths?: Record<string, string>;
  evidence_paths?: Record<string, string>;
  runner_warnings?: string[];
  metadata: JobSummary;
  progress?: JobProgress;
  stdout_tail: string[];
  stderr_tail: string[];
  can_resume: boolean;
}

export interface JobStreamEvent {
  event: "snapshot" | "progress" | "stdout" | "stderr" | "status" | "done" | "error" | string;
  job_id: string;
  status: string;
  timestamp?: string;
  stream?: "stdout" | "stderr" | string;
  line?: string;
  line_no?: number;
  metadata?: JobSummary & { progress?: JobProgress; provider_health?: ProviderHealthPayload };
  progress?: JobProgress;
  message?: string;
}

export interface JobStreamSubscription {
  close(): void;
}

export interface JobStreamHandlers {
  onEvent?: (event: JobStreamEvent) => void;
  onError?: (error: Event | Error) => void;
  onOpen?: () => void;
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

export interface DoctorPayload {
  status: string;
  checked_at: string;
  checks: Array<{ name: string; ok: boolean; detail: string }>;
}

export interface ExecutionApi {
  getDailyRunStatus(): Promise<DailyRunStatusPayload>;
  getLatestDailyRun(): Promise<DailyRunVerdict>;
  getDataReadiness(candidateDate?: string): Promise<DataReadinessPayload>;
  getSystemDoctor(): Promise<DoctorPayload>;
  getDoctor(): Promise<DoctorPayload>;
  getModels(): Promise<ModelsPayload>;
  getModelDetail(modelId: string): Promise<ModelDetailPayload>;
  trainModel(modelId: string, payload: TrainModelRequest): Promise<JobLaunchPayload>;
  getDataSources(): Promise<DataSourcesPayload>;
  refreshDataSources(payload: DataRefreshRequest): Promise<JobLaunchPayload>;
  runProviderHealth(payload: ProviderHealthRequest): Promise<JobLaunchPayload>;
  getTradePlan(): Promise<TradePlanPayload>;
  generateTradePlan(payload: TradePlanGenerateRequest): Promise<JobLaunchPayload>;
  getAccount(): Promise<AccountPayload>;
  saveAccount(payload: AccountSaveRequest): Promise<AccountPayload>;
  resetAccountExample(): Promise<AccountPayload>;
  getPaperAccount(): Promise<PaperAccountPayload>;
  recordPaperCashFlow(payload: PaperCashFlowRequest): Promise<PaperAccountPayload>;
  recordPaperManualAdjustment(payload: PaperManualAdjustmentRequest): Promise<PaperAccountPayload>;
  applyLatestPaperPlan(payload?: { execution_date?: string }): Promise<JsonObject>;
  getPaperPerformance(startDate: string, endDate: string): Promise<PaperPerformancePayload>;
  getJobs(limit?: number): Promise<JobSummary[]>;
  getJob(jobId: string, lines?: number): Promise<JobDetail>;
  streamJob?: (jobId: string, handlers: JobStreamHandlers) => JobStreamSubscription;
  resumeJob(jobId: string): Promise<JobLaunchPayload>;
  unlockRuntime(force: boolean): Promise<{ status: string; detail: string }>;
}
