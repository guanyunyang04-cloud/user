# Daily Stock Analysis Environment Model

## 1. 语言与框架
- Python：
  - `3.10+`
- 后端：
  - `FastAPI`
  - `uvicorn`
- Web / Desktop：
  - `apps/dsa-web`
  - `apps/dsa-desktop`

## 2. 常用入口
```bash
python main.py
python main.py --debug
python main.py --dry-run
python main.py --serve
python main.py --serve-only
uvicorn server:app --reload --host 0.0.0.0 --port 8000
python webui.py
```

## 3. 常用验证
```bash
pip install -r requirements.txt
./scripts/ci_gate.sh
python -m pytest -m "not network"
```

## 4. AI 协作补充
- 仓库内 AI 协作规则真源：
  - `daily_stock_analysis-main/AGENTS.md`
- 当前工作区级主脑：
  - `brain/master_brain.md`
