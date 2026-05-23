import { useEffect, useMemo, useState } from "react";
import { Play, RefreshCw, ShieldAlert } from "lucide-react";
import type { ExecutionApi, ModelCard, ModelDetailPayload, ModelsPayload, TrainModelRequest } from "../types";
import { DataTable, ErrorState, Field, LoadingState, PageHeader, Panel, Stat, StatusPill } from "../components";
import { text } from "../format";

interface ModelsPageProps {
  api: ExecutionApi;
}

const ROLE_LABELS: Record<string, string> = {
  live: "Live",
  production: "Production",
  path_policy: "Research",
  legacy: "Legacy"
};

function isProduction(model: ModelCard | null): boolean {
  return Boolean(model?.role === "production" || model?.id.includes("production"));
}

export function ModelsPage({ api }: ModelsPageProps): JSX.Element {
  const [payload, setPayload] = useState<ModelsPayload | null>(null);
  const [selectedId, setSelectedId] = useState("");
  const [detail, setDetail] = useState<ModelDetailPayload | null>(null);
  const [loading, setLoading] = useState(true);
  const [detailLoading, setDetailLoading] = useState(false);
  const [error, setError] = useState("");
  const [datasetMode, setDatasetMode] = useState<TrainModelRequest["dataset_mode"]>("latest");
  const [datasetId, setDatasetId] = useState("");
  const [startDate, setStartDate] = useState("");
  const [endDate, setEndDate] = useState("");
  const [advancedArgs, setAdvancedArgs] = useState("");
  const [confirmedProduction, setConfirmedProduction] = useState(false);
  const [jobMessage, setJobMessage] = useState("");
  const [submitting, setSubmitting] = useState(false);

  useEffect(() => {
    let alive = true;
    setLoading(true);
    api
      .getModels()
      .then((next) => {
        if (!alive) return;
        setPayload(next);
        setSelectedId((current) => current || next.models[0]?.id || "");
        setError("");
      })
      .catch((err: Error) => alive && setError(err.message))
      .finally(() => alive && setLoading(false));
    return () => {
      alive = false;
    };
  }, [api]);

  useEffect(() => {
    if (!selectedId) return;
    let alive = true;
    setDetailLoading(true);
    api
      .getModelDetail(selectedId)
      .then((next) => {
        if (!alive) return;
        setDetail(next);
        setError("");
      })
      .catch((err: Error) => alive && setError(err.message))
      .finally(() => alive && setDetailLoading(false));
    return () => {
      alive = false;
    };
  }, [api, selectedId]);

  const selectedModel = useMemo(
    () => payload?.models.find((item) => item.id === selectedId) || null,
    [payload, selectedId]
  );
  const production = isProduction(selectedModel);
  const trainDisabled = submitting || (production && !confirmedProduction);

  async function submitTrain(): Promise<void> {
    if (!selectedModel) return;
    setSubmitting(true);
    setJobMessage("");
    try {
      const response = await api.trainModel(selectedModel.id, {
        dataset_mode: datasetMode,
        dataset_id: datasetMode === "dataset_id" ? datasetId : "",
        start_date: startDate,
        end_date: endDate,
        advanced_args: advancedArgs,
        force_unlock: false
      });
      setJobMessage(`已提交作业 ${response.job_id || ""} (${response.status})`);
    } catch (err) {
      setJobMessage(err instanceof Error ? err.message : "提交失败");
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div>
      <PageHeader
        title="模型"
        eyebrow="集中查看 live、production、research、legacy"
        actions={<button onClick={() => api.getModels().then(setPayload)}><RefreshCw size={16} />刷新</button>}
      />
      {loading ? <LoadingState /> : null}
      {error ? <ErrorState message={error} /> : null}
      <div className="two-column">
        <Panel title="模型列表" meta={payload?.updated_at || ""}>
          <div className="model-list">
            {(payload?.models || []).map((model) => (
              <button
                key={model.id}
                className={`model-row ${model.id === selectedId ? "is-active" : ""}`}
                onClick={() => {
                  setSelectedId(model.id);
                  setConfirmedProduction(false);
                }}
              >
                <span>
                  <strong>{model.name}</strong>
                  <small>{ROLE_LABELS[model.role] || model.role} / {model.usage}</small>
                </span>
                <StatusPill value={model.status} />
              </button>
            ))}
          </div>
        </Panel>

        <Panel title={selectedModel ? selectedModel.name : "模型详情"}>
          {detailLoading ? <LoadingState label="加载模型详情" /> : null}
          {selectedModel ? (
            <>
              <div className="stat-grid">
                <Stat label="角色" value={ROLE_LABELS[selectedModel.role] || selectedModel.role} />
                <Stat label="数据源" value={text(selectedModel.data_source)} />
                <Stat label="训练时间" value={text(selectedModel.trained_at)} />
                <Stat label="样本截止" value={text(selectedModel.train_end_date)} />
                <Stat label="信号日" value={text(selectedModel.signal_date)} />
                <Stat label="Dataset" value={text(selectedModel.dataset_id)} />
              </div>
              <div className="detail-json">
                <pre>{JSON.stringify(detail?.model?.detail || selectedModel.detail || {}, null, 2)}</pre>
              </div>
            </>
          ) : null}
        </Panel>
      </div>

      <Panel
        title="显式训练 / 重训"
        meta={production ? <span className="danger-inline"><ShieldAlert size={15} />production 需要确认</span> : "默认 research"}
      >
        <div className="form-grid">
          <div className="segmented" role="radiogroup" aria-label="数据集模式">
            <label>
              <input
                type="radio"
                name="dataset_mode"
                checked={datasetMode === "latest"}
                onChange={() => setDatasetMode("latest")}
              />
              最新数据集
            </label>
            <label>
              <input
                type="radio"
                name="dataset_mode"
                checked={datasetMode === "dataset_id"}
                onChange={() => setDatasetMode("dataset_id")}
              />
              指定数据集
            </label>
            <label>
              <input
                type="radio"
                name="dataset_mode"
                checked={datasetMode === "custom"}
                onChange={() => setDatasetMode("custom")}
              />
              自定义时间
            </label>
          </div>
          <Field label="Dataset ID">
            <input value={datasetId} onChange={(event) => setDatasetId(event.target.value)} disabled={datasetMode !== "dataset_id"} />
          </Field>
          <Field label="训练开始">
            <input value={startDate} onChange={(event) => setStartDate(event.target.value)} placeholder="20250101" />
          </Field>
          <Field label="训练结束">
            <input value={endDate} onChange={(event) => setEndDate(event.target.value)} placeholder="20260522" />
          </Field>
          <Field label="高级参数">
            <input value={advancedArgs} onChange={(event) => setAdvancedArgs(event.target.value)} placeholder="显式附加参数" />
          </Field>
          {production ? (
            <label className="confirm-line">
              <input
                type="checkbox"
                checked={confirmedProduction}
                onChange={(event) => setConfirmedProduction(event.target.checked)}
              />
              确认 production 重训
            </label>
          ) : null}
        </div>
        <div className="command-note">
          将提交到 <code>{selectedModel?.id || "-"}</code>；不会自动 promotion 或改写 active manifest，除非底层任务本身显式支持且你传入对应参数。
        </div>
        <button className="primary" disabled={trainDisabled || !selectedModel} onClick={submitTrain}>
          <Play size={16} />启动训练
        </button>
        {jobMessage ? <p className="inline-message">{jobMessage}</p> : null}
      </Panel>

      <Panel title="模型卡片原始表">
        <DataTable
          rows={(payload?.models || []).map((model) => ({
            id: model.id,
            role: model.role,
            status: model.status,
            train_end_date: model.train_end_date,
            dataset_id: model.dataset_id,
            artifact_path: model.artifact_path
          }))}
          preferredColumns={["id", "role", "status", "train_end_date", "dataset_id", "artifact_path"]}
        />
      </Panel>
    </div>
  );
}
