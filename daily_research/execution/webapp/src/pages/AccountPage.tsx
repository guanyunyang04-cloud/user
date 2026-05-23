import { useEffect, useState } from "react";
import { Plus, RotateCcw, Save, Trash2 } from "lucide-react";
import type { AccountPayload, AccountPosition, ExecutionApi } from "../types";
import { ErrorState, Field, LoadingState, PageHeader, Panel, Stat } from "../components";
import { text } from "../format";

interface AccountPageProps {
  api: ExecutionApi;
}

export function AccountPage({ api }: AccountPageProps): JSX.Element {
  const [payload, setPayload] = useState<AccountPayload | null>(null);
  const [cash, setCash] = useState("");
  const [positions, setPositions] = useState<AccountPosition[]>([]);
  const [loading, setLoading] = useState(true);
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");

  const load = (): void => {
    setLoading(true);
    api
      .getAccount()
      .then((next) => {
        setPayload(next);
        setCash(next.available_cash === null ? "" : String(next.available_cash));
        setPositions(next.positions || []);
        setError("");
      })
      .catch((err: Error) => setError(err.message))
      .finally(() => setLoading(false));
  };

  useEffect(() => {
    load();
  }, []);

  function updatePosition(index: number, patch: Partial<AccountPosition>): void {
    setPositions((current) => current.map((item, itemIndex) => (itemIndex === index ? { ...item, ...patch } : item)));
  }

  async function save(): Promise<void> {
    setMessage("");
    try {
      const next = await api.saveAccount({ available_cash: cash, positions });
      setPayload(next);
      setMessage("账户已保存");
    } catch (err) {
      setError(err instanceof Error ? err.message : "保存失败");
    }
  }

  async function reset(): Promise<void> {
    setMessage("");
    try {
      const next = await api.resetAccountExample();
      setPayload(next);
      setCash(next.available_cash === null ? "" : String(next.available_cash));
      setPositions(next.positions || []);
      setMessage("已从示例账户恢复");
    } catch (err) {
      setError(err instanceof Error ? err.message : "恢复失败");
    }
  }

  return (
    <div>
      <PageHeader title="账户" eyebrow="当前模拟账户与持仓输入" actions={<button onClick={reset}><RotateCcw size={16} />示例恢复</button>} />
      {loading ? <LoadingState /> : null}
      {error ? <ErrorState message={error} /> : null}
      <div className="stat-grid">
        <Stat label="文件" value={text(payload?.path)} />
        <Stat label="现金" value={text(payload?.available_cash)} />
        <Stat label="持仓数" value={payload?.position_count || 0} />
        <Stat label="更新时间" value={text(payload?.last_modified_at)} />
      </div>
      <Panel title="账户编辑">
        <Field label="可用现金">
          <input value={cash} onChange={(event) => setCash(event.target.value)} />
        </Field>
        <div className="positions-editor">
          <div className="positions-head">
            <span>证券代码</span>
            <span>数量</span>
            <span>成本价</span>
            <span />
          </div>
          {positions.map((position, index) => (
            <div className="positions-row" key={`${position.stock}-${index}`}>
              <input value={position.stock} onChange={(event) => updatePosition(index, { stock: event.target.value })} />
              <input value={position.shares} onChange={(event) => updatePosition(index, { shares: event.target.value })} />
              <input value={position.cost_price} onChange={(event) => updatePosition(index, { cost_price: event.target.value })} />
              <button onClick={() => setPositions((current) => current.filter((_, itemIndex) => itemIndex !== index))} aria-label={`删除 ${position.stock || index}`}>
                <Trash2 size={15} />
              </button>
            </div>
          ))}
        </div>
        <div className="button-row">
          <button onClick={() => setPositions((current) => [...current, { stock: "", shares: "", cost_price: "" }])}><Plus size={16} />添加持仓</button>
          <button className="primary" onClick={save}><Save size={16} />保存账户</button>
        </div>
        {message ? <p className="inline-message">{message}</p> : null}
      </Panel>
    </div>
  );
}
