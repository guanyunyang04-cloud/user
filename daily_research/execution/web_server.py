from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn

from daily_research.execution import app_service
from daily_research.execution.web_models import (
    AccountSnapshotRequest,
    DataRefreshRequest,
    ModelTrainRequest,
    PaperApplyLatestPlanRequest,
    PaperCashFlowRequest,
    PaperManualAdjustmentRequest,
    ProviderHealthRequest,
    ResumeRequest,
    SchedulerConfigRequest,
    TaskRunRequest,
    TradePlanGenerateRequest,
    UnlockRequest,
)
from daily_research.execution.web_service import (
    account_context,
    base_context,
    dashboard_context,
    doctor_context,
    guide_context,
    job_detail_context,
    jobs_context,
    runtime_context,
    section_label,
    status_label,
    tasks_context,
    trade_plan_context,
    ui_paths,
)


UI_PATHS = ui_paths()
TEMPLATES = Jinja2Templates(directory=str(UI_PATHS["templates"]))
TEMPLATES.env.filters["status_label"] = status_label
TEMPLATES.env.filters["section_label"] = section_label
RETIRED_FRONTEND_PATHS = {"/continuous-policy"}


def _react_index_response() -> FileResponse | None:
    index_path = UI_PATHS.get("react_dist", UI_PATHS["ui_root"]) / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return None


def _render_template(
    request: Request,
    template_name: str,
    *,
    active_path: str,
    context: dict[str, Any] | None = None,
) -> HTMLResponse:
    payload = base_context(active_path=active_path)
    payload.update(context if isinstance(context, dict) else {})
    payload["request"] = request
    return TEMPLATES.TemplateResponse(request=request, name=template_name, context=payload)


def create_app() -> FastAPI:
    app = FastAPI(
        title="Daily Research 执行控制台",
        version="1.0.0",
        docs_url="/api/docs",
        redoc_url=None,
    )
    app.mount("/static", StaticFiles(directory=str(UI_PATHS["static"])), name="static")
    react_dist = UI_PATHS.get("react_dist")
    if react_dist is not None and react_dist.exists():
        assets_dir = react_dist / "assets"
        if assets_dir.exists():
            app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="react-assets")

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/", response_class=HTMLResponse)
    def dashboard_page(request: Request) -> Response:
        react_response = _react_index_response()
        if react_response is not None:
            return react_response
        return _render_template(request, "dashboard.html", active_path="/", context=dashboard_context())

    @app.get("/tasks", response_class=HTMLResponse)
    def tasks_page(request: Request, task: str = Query(default="")) -> Response:
        react_response = _react_index_response()
        if react_response is not None:
            return react_response
        return _render_template(request, "tasks.html", active_path="/tasks", context=tasks_context(selected_task=task))

    @app.get("/jobs", response_class=HTMLResponse)
    def jobs_page(request: Request, limit: int = Query(default=30, ge=1, le=100)) -> Response:
        react_response = _react_index_response()
        if react_response is not None:
            return react_response
        return _render_template(request, "jobs.html", active_path="/jobs", context=jobs_context(limit=limit))

    @app.get("/jobs/{job_id}", response_class=HTMLResponse)
    def job_page(request: Request, job_id: str, lines: int = Query(default=120, ge=10, le=400)) -> Response:
        react_response = _react_index_response()
        if react_response is not None:
            return react_response
        try:
            context = job_detail_context(job_id, lines=lines)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return _render_template(request, "job_detail.html", active_path="/jobs", context=context)

    @app.get("/doctor", response_class=HTMLResponse)
    def doctor_page(request: Request) -> Response:
        react_response = _react_index_response()
        if react_response is not None:
            return react_response
        return _render_template(request, "doctor.html", active_path="/doctor", context=doctor_context())

    @app.get("/artifacts/trade-plan", response_class=HTMLResponse)
    def trade_plan_page(request: Request) -> Response:
        react_response = _react_index_response()
        if react_response is not None:
            return react_response
        return _render_template(request, "trade_plan.html", active_path="/artifacts/trade-plan", context=trade_plan_context())

    @app.get("/account", response_class=HTMLResponse)
    def account_page(request: Request) -> Response:
        react_response = _react_index_response()
        if react_response is not None:
            return react_response
        return _render_template(request, "account.html", active_path="/account", context=account_context())

    @app.get("/guide", response_class=HTMLResponse)
    def guide_page(request: Request) -> Response:
        react_response = _react_index_response()
        if react_response is not None:
            return react_response
        return _render_template(request, "guide.html", active_path="/guide", context=guide_context())

    @app.get("/settings/runtime", response_class=HTMLResponse)
    def runtime_page(request: Request) -> Response:
        react_response = _react_index_response()
        if react_response is not None:
            return react_response
        return _render_template(request, "runtime.html", active_path="/settings/runtime", context=runtime_context())

    @app.get("/api/status")
    def api_status(history_limit: int = Query(default=10, ge=1, le=100)) -> dict[str, Any]:
        return app_service.build_status_payload(history_limit=history_limit)

    @app.get("/api/doctor")
    def api_doctor() -> dict[str, Any]:
        return app_service.build_doctor_payload()

    @app.get("/api/tasks")
    def api_tasks(core_only: bool = Query(default=True)) -> list[dict[str, Any]]:
        return app_service.list_tasks_payload(core_only=core_only)

    @app.get("/api/tasks/{task_name}")
    def api_task(task_name: str) -> dict[str, Any]:
        try:
            return next(item for item in app_service.list_tasks_payload(core_only=False) if item["name"] == task_name)
        except StopIteration as exc:
            raise HTTPException(status_code=404, detail=f"未找到任务：{task_name}") from exc

    @app.get("/api/jobs")
    def api_jobs(limit: int = Query(default=20, ge=1, le=100)) -> list[dict[str, Any]]:
        return app_service.list_jobs_payload(limit=limit)

    @app.get("/api/jobs/{job_id}")
    def api_job(job_id: str, lines: int = Query(default=120, ge=10, le=400)) -> dict[str, Any]:
        try:
            return app_service.build_job_detail_payload(job_id, lines=lines)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/jobs/{job_id}/tail")
    def api_job_tail(job_id: str, lines: int = Query(default=120, ge=10, le=400)) -> dict[str, Any]:
        try:
            payload = app_service.build_job_detail_payload(job_id, lines=lines)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return {
            "job_id": payload["job_id"],
            "status": payload["status"],
            "stdout_tail": payload["stdout_tail"],
            "stderr_tail": payload["stderr_tail"],
        }

    @app.get("/api/trade-plan")
    def api_trade_plan() -> dict[str, Any]:
        return app_service.latest_trade_plan_summary(max_lines=240)

    @app.post("/api/trade-plan/generate")
    def api_generate_trade_plan(request: TradePlanGenerateRequest) -> JSONResponse:
        form_payload: dict[str, Any] = {}
        if request.candidate_profile:
            form_payload["candidate_profile"] = request.candidate_profile
        raw_args: list[str] = []
        if request.positions_file:
            raw_args.extend(["--positions-file", request.positions_file])
        if request.cash not in {None, ""}:
            raw_args.extend(["--cash", str(request.cash)])
        if request.lot_size not in {None, ""}:
            raw_args.extend(["--lot-size", str(request.lot_size)])
        if request.target_weight_top_k not in {None, ""}:
            raw_args.extend(["--target-weight-top-k", str(request.target_weight_top_k)])
        if request.target_weight_min_weight not in {None, ""}:
            raw_args.extend(["--target-weight-min-weight", str(request.target_weight_min_weight)])
        if request.raw_args_text:
            raw_args.extend(app_service.parse_raw_args_text(request.raw_args_text))
        try:
            payload = app_service.launch_task_async(
                task_name="trade-plan",
                passthrough_args=[
                    *app_service.build_passthrough_args_from_form(
                        task_name="trade-plan",
                        form_payload=form_payload,
                    ),
                    *raw_args,
                ],
                job_label=request.job_label,
                force_unlock=request.force_unlock,
            )
            return JSONResponse(payload)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/models")
    def api_models() -> dict[str, Any]:
        return app_service.models_summary()

    @app.get("/api/models/{model_id}")
    def api_model(model_id: str) -> dict[str, Any]:
        try:
            return app_service.model_detail(model_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/api/models/{model_id}/train")
    def api_model_train(model_id: str, request: ModelTrainRequest) -> JSONResponse:
        try:
            if model_id != "production-full-fit":
                raise ValueError(f"模型 {model_id} 暂未接入执行端训练入口。")
            task_name = "refresh-production-default"
            raw_args: list[str] = []
            if request.start_date:
                raw_args.extend(["--start-date", request.start_date])
            if request.end_date:
                raw_args.extend(["--end-date", request.end_date])
            if request.advanced_args:
                raw_args.extend(app_service.parse_raw_args_text(request.advanced_args))
            payload = app_service.launch_task_async(
                task_name=task_name,
                passthrough_args=raw_args,
                job_label=request.job_label or f"manual-train:{model_id}",
                force_unlock=request.force_unlock,
            )
            return JSONResponse(payload)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/data-sources")
    def api_data_sources() -> dict[str, Any]:
        return app_service.data_sources_summary()

    @app.post("/api/data-sources/provider-health")
    def api_data_sources_provider_health(request: ProviderHealthRequest) -> JSONResponse:
        try:
            payload = app_service.provider_health_summary(
                provider_plan=request.provider_plan or app_service.FORMAL_DATA_PLATFORM_PROVIDER_PLAN,
                as_of_date=request.as_of_date,
                domains=request.domains or list(app_service.FORMAL_DATA_PLATFORM_REQUIRED_DOMAINS),
                symbols=request.symbols or ["000001.SZ", "600000.SH", "000300.SH"],
            )
            return JSONResponse(payload)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/data-sources/scheduler")
    def api_data_sources_scheduler() -> dict[str, Any]:
        return app_service.scheduler_summary()

    @app.patch("/api/data-sources/scheduler")
    def api_data_sources_scheduler_update(request: SchedulerConfigRequest) -> JSONResponse:
        try:
            patch: dict[str, Any] = {}
            if request.enabled is not None:
                patch["enabled"] = request.enabled
            if request.post_close_time:
                patch["post_close_time"] = request.post_close_time
            return JSONResponse(app_service.update_scheduler_config(patch))
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/data-sources/refresh")
    def api_data_sources_refresh(request: DataRefreshRequest) -> JSONResponse:
        try:
            default_domains = list(app_service.FORMAL_DATA_PLATFORM_DOMAINS)
            as_of_date = str(request.as_of_date or "").strip() or str(app_service.get_latest_completed_trading_date())
            universe = str(request.universe or "").strip() or "all_a"
            domains = [str(item).strip() for item in (request.domains or []) if str(item).strip()] or default_domains
            current_data = app_service.data_sources_summary()
            default_as_of = str(
                current_data.get("data_platform", {})
                .get("default_refresh", {})
                .get("as_of_date", "")
                or current_data.get("data_platform", {}).get("latest_completed_trading_date", "")
                or as_of_date
            )
            requested_is_default = (
                not str(request.start_date or "").strip()
                and universe == "all_a"
                and [item for item in domains] == default_domains
                and str(as_of_date) == default_as_of
            )
            if (
                requested_is_default
                and current_data.get("next_refresh_action") == "skip"
                and current_data.get("next_signal_action") == "refresh"
            ):
                payload = app_service.launch_task_async(
                    task_name="refresh-production-live-panels",
                    passthrough_args=["--as-of-date", str(as_of_date)],
                    job_label=request.job_label or "manual-signal-refresh",
                    force_unlock=request.force_unlock,
                )
                return JSONResponse(payload)
            if requested_is_default and current_data.get("next_refresh_action") == "skip":
                return JSONResponse(app_service.data_refresh_skip_payload(data_sources=current_data, as_of_date=as_of_date))
            args: list[str] = []
            if as_of_date:
                args.extend(["--as-of-date", as_of_date])
            if request.start_date:
                args.extend(["--start-date", request.start_date])
            if universe:
                args.extend(["--universe", universe])
            args.extend(["--provider-plan", app_service.FORMAL_DATA_PLATFORM_PROVIDER_PLAN])
            args.extend(["--required-domains", ",".join(app_service.FORMAL_DATA_PLATFORM_REQUIRED_DOMAINS)])
            if domains:
                args.extend(["--domains", ",".join(domains)])
            if request.advanced_args:
                advanced_args = app_service.strip_execution_timeout_args(app_service.parse_raw_args_text(request.advanced_args))
                args.extend(advanced_args)
            payload = app_service.launch_task_async(
                task_name="data-platform-refresh",
                passthrough_args=args,
                job_label=request.job_label or "manual-data-refresh",
                force_unlock=request.force_unlock,
            )
            return JSONResponse(payload)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/paper-account")
    def api_paper_account() -> dict[str, Any]:
        return app_service.paper_account_summary()

    @app.post("/api/paper-account/cash-flow")
    def api_paper_account_cash_flow(request: PaperCashFlowRequest) -> JSONResponse:
        try:
            payload = app_service.paper_account_cash_flow(
                flow_type=request.flow_type,
                amount=request.amount,
                reason=request.reason,
            )
            return JSONResponse(payload)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/paper-account/manual-adjustment")
    def api_paper_account_manual_adjustment(request: PaperManualAdjustmentRequest) -> JSONResponse:
        try:
            payload = app_service.paper_account_manual_adjustment(
                adjustment_type=request.adjustment_type,
                stock=request.stock,
                shares=request.shares,
                cost_price=request.cost_price,
                amount=request.amount,
                reason=request.reason,
            )
            return JSONResponse(payload)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/paper-account/apply-latest-plan")
    def api_paper_account_apply_latest_plan(request: PaperApplyLatestPlanRequest) -> JSONResponse:
        try:
            payload = app_service.paper_account_apply_latest_plan(execution_date=request.execution_date)
            return JSONResponse(payload)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/paper-account/performance")
    def api_paper_account_performance(
        start_date: str = Query(default=""),
        end_date: str = Query(default=""),
    ) -> dict[str, Any]:
        if not start_date or not end_date:
            raise HTTPException(status_code=400, detail="start_date and end_date are required")
        return app_service.paper_account_performance(start_date=start_date, end_date=end_date)

    @app.get("/api/account")
    def api_account() -> dict[str, Any]:
        return app_service.load_account_snapshot()

    @app.post("/api/account")
    def api_save_account(request: AccountSnapshotRequest) -> JSONResponse:
        try:
            payload = app_service.save_account_snapshot(
                available_cash=request.available_cash,
                positions=[item.model_dump() for item in request.positions],
            )
            return JSONResponse(payload)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/account/reset-example")
    def api_reset_account_example() -> JSONResponse:
        try:
            payload = app_service.reset_account_snapshot_from_example()
            return JSONResponse(payload)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/run")
    def api_run(request: TaskRunRequest) -> JSONResponse:
        try:
            passthrough_args = app_service.build_passthrough_args_from_form(
                task_name=request.task_name,
                form_payload=request.form_payload,
                raw_args_text=request.raw_args_text,
            )
            if request.background:
                payload = app_service.launch_task_async(
                    task_name=request.task_name,
                    python_executable=request.python_executable,
                    passthrough_args=passthrough_args,
                    job_label=request.job_label,
                    force_unlock=request.force_unlock,
                )
                return JSONResponse(payload)
            payload = app_service.run_task_sync(
                task_name=request.task_name,
                python_executable=request.python_executable,
                passthrough_args=passthrough_args,
                job_label=request.job_label,
                force_unlock=request.force_unlock,
                echo_output=False,
            )
            return JSONResponse(payload)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/resume")
    def api_resume(request: ResumeRequest) -> JSONResponse:
        try:
            if request.background:
                payload = app_service.resume_task_async(
                    job_id=request.job_id,
                    job_label=request.job_label,
                    force_unlock=request.force_unlock,
                )
                return JSONResponse(payload)
            payload = app_service.resume_task_sync(
                job_id=request.job_id,
                job_label=request.job_label,
                force_unlock=request.force_unlock,
                echo_output=False,
            )
            return JSONResponse(payload)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/unlock")
    def api_unlock(request: UnlockRequest) -> JSONResponse:
        try:
            return JSONResponse(app_service.unlock_runtime(force=request.force))
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/{full_path:path}", response_class=HTMLResponse, response_model=None)
    def react_app_fallback(full_path: str) -> Response:
        request_path = f"/{str(full_path).strip('/')}"
        if request_path in RETIRED_FRONTEND_PATHS:
            raise HTTPException(status_code=404, detail=f"Retired execution console path: {request_path}")
        if str(full_path).startswith("api/"):
            raise HTTPException(status_code=404, detail=f"API path not found: /{full_path}")
        react_response = _react_index_response()
        if react_response is not None:
            return react_response
        raise HTTPException(status_code=404, detail=f"React app build not found for path: /{full_path}")

    return app


app = create_app()


def run_web_console(*, host: str = "127.0.0.1", port: int = 8765, reload: bool = False) -> None:
    if not reload:
        app_service.start_scheduler_loop()
    target = "daily_research.execution.web_server:app" if reload else app
    uvicorn.run(target, host=str(host), port=int(port), reload=bool(reload), log_level="info")
