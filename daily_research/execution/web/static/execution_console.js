(function () {
  const STATUS_LABELS = {
    ok: "正常",
    queued: "排队中",
    running: "运行中",
    succeeded: "成功",
    failed: "失败",
    blocked: "阻塞",
    degraded: "降级",
  };

  function $(selector) {
    return document.querySelector(selector);
  }

  function escapeHtml(value) {
    return String(value ?? "")
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#39;");
  }

  function statusClass(status) {
    const normalized = String(status || "").toLowerCase();
    if (["succeeded", "ok"].includes(normalized)) return "status-succeeded";
    if (["running", "queued"].includes(normalized)) return "status-running";
    if (["failed", "blocked", "degraded"].includes(normalized)) return "status-failed";
    return "status-running";
  }

  function statusText(status) {
    const normalized = String(status || "").toLowerCase();
    return STATUS_LABELS[normalized] || String(status || "未知");
  }

  async function fetchJson(url, options) {
    const response = await fetch(url, options);
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || payload.error || response.statusText);
    }
    return payload;
  }

  function renderJobsRows(rows, options) {
    const includeLabel = Boolean(options && options.includeLabel);
    return (rows || [])
      .map(
        (job) => `
          <tr>
            <td><a href="/jobs/${escapeHtml(job.job_id)}">${escapeHtml(job.job_id)}</a></td>
            <td>${escapeHtml(job.task_name)}</td>
            <td><span class="status-pill ${statusClass(job.status)}">${escapeHtml(statusText(job.status))}</span></td>
            ${includeLabel ? `<td>${escapeHtml(job.job_label || "未填写")}</td>` : ""}
            <td>${escapeHtml(job.started_at || "暂无")}</td>
            <td>${escapeHtml(job.completed_at || "暂无")}</td>
          </tr>
        `
      )
      .join("");
  }

  function startPolling(fn, intervalMs) {
    fn();
    return window.setInterval(fn, intervalMs);
  }

  function updateText(selector, value) {
    const node = $(selector);
    if (node) node.textContent = value ?? "";
  }

  function initDashboard() {
    const refresh = async () => {
      const payload = await fetchJson("/api/status?history_limit=10");
      updateText("#summary-current-job", payload.current_job?.job_id || "空闲");
      updateText("#summary-lock", payload.lock?.job_id || "未锁定");
      updateText("#summary-yolos", payload.yolos_python || "暂无");
      updateText("#summary-updated", payload.updated_at || "暂无");
      updateText("#manifest-strategy", payload.active_manifest?.strategy_name || "暂无");
      updateText("#manifest-candidate", payload.active_manifest?.candidate_label || "暂无");
      updateText("#manifest-pool", payload.active_manifest?.liquidity_pool_name || "暂无");
      updateText("#manifest-profile", payload.active_manifest?.effective_live_execution_profile || "暂无");
      updateText("#manifest-live-mode", payload.active_manifest?.effective_live_target_weight_mode || "暂无");
      updateText("#manifest-trade-panel", payload.active_manifest?.trade_plan_target_weight_panel_csv || "暂无");
      updateText("#positions-row-count", String(payload.current_positions?.row_count ?? 0));
      updateText(
        "#positions-cash",
        payload.current_positions?.available_cash !== null && payload.current_positions?.available_cash !== undefined
          ? String(payload.current_positions.available_cash)
          : "暂无"
      );
      updateText("#positions-headers", payload.current_positions?.headers_ok ? "正常" : "异常");
      updateText("#positions-tickers", (payload.current_positions?.tickers || []).join(", ") || "暂无");
      updateText("#positions-updated", payload.current_positions?.last_modified_at || "暂无");
      const jobsBody = $("#recent-jobs-body");
      if (jobsBody) jobsBody.innerHTML = renderJobsRows(payload.recent_jobs || [], { includeLabel: false });
      const preview = $("#trade-plan-preview");
      if (preview) preview.textContent = (payload.latest_trade_plan?.preview_lines || []).join("\n");
      const doctorPayload = await fetchJson("/api/doctor");
      const doctorChip = $("#doctor-status-chip");
      if (doctorChip) {
        doctorChip.textContent = statusText(doctorPayload.status);
        doctorChip.className = `signal-chip ${doctorPayload.status === "ok" ? "is-good" : "is-warn"}`;
      }
      updateText("#doctor-summary-text", `已完成 ${(doctorPayload.checks || []).length} 项检查。`);
      const doctorList = $("#doctor-checks-list");
      if (doctorList) {
        doctorList.innerHTML = (doctorPayload.checks || [])
          .slice(0, 6)
          .map(
            (check) => `
              <li class="${check.ok ? "is-good" : "is-bad"}">
                <span>${escapeHtml(check.name)}</span>
                <small>${escapeHtml(check.detail)}</small>
              </li>
            `
          )
          .join("");
      }
    };
    startPolling(refresh, 4000);
  }

  function initJobs() {
    const refresh = async () => {
      const payload = await fetchJson("/api/jobs?limit=30");
      const body = $("#jobs-table-body");
      if (body) body.innerHTML = renderJobsRows(payload, { includeLabel: true });
    };
    startPolling(refresh, 3500);
  }

  function formPayloadFromForm(form) {
    const payload = {};
    form.querySelectorAll("[name]").forEach((element) => {
      const name = element.getAttribute("name");
      if (!name || ["job_label", "raw_args_text"].includes(name)) return;
      if (element.type === "checkbox") {
        payload[name] = element.checked;
        return;
      }
      payload[name] = element.value;
    });
    return payload;
  }

  function showLaunchResult(message, isError, jobId) {
    const result = $("#task-launch-result");
    if (!result) return;
    result.innerHTML = `
      <p>${escapeHtml(message)}</p>
      ${jobId ? `<p><a href="/jobs/${escapeHtml(jobId)}">打开作业 ${escapeHtml(jobId)}</a></p>` : ""}
    `;
    result.style.borderColor = isError ? "rgba(181, 58, 47, 0.35)" : "rgba(15, 123, 99, 0.35)";
  }

  function initTasks() {
    const form = $("#task-launch-form");
    const launcher = $(".task-launcher");
    if (!form || !launcher) return;
    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      const taskName = launcher.dataset.taskName;
      const jobLabel = form.querySelector('[name="job_label"]')?.value || "";
      const rawArgsText = form.querySelector('[name="raw_args_text"]')?.value || "";
      showLaunchResult("正在提交任务...", false, "");
      try {
        const payload = await fetchJson("/api/run", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            task_name: taskName,
            job_label: jobLabel,
            form_payload: formPayloadFromForm(form),
            raw_args_text: rawArgsText,
            background: true,
          }),
        });
        showLaunchResult(`任务已提交，作业号为 ${payload.job_id}。`, false, payload.job_id);
        window.setTimeout(() => {
          window.location.href = `/jobs/${payload.job_id}`;
        }, 700);
      } catch (error) {
        showLaunchResult(error.message || "任务提交失败。", true, "");
      }
    });
  }

  function initJobDetail() {
    const rerunButton = $("#job-rerun-button");
    const resumeButton = $("#job-resume-button");
    const jobId = rerunButton?.dataset.jobId || resumeButton?.dataset.jobId;
    if (!jobId) return;

    async function triggerResume(labelPrefix) {
      const payload = await fetchJson("/api/resume", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          job_id: jobId,
          job_label: `${labelPrefix}:${jobId}`,
          background: true,
        }),
      });
      window.location.href = `/jobs/${payload.job_id}`;
    }

    if (rerunButton) rerunButton.addEventListener("click", () => triggerResume("重跑"));
    if (resumeButton) resumeButton.addEventListener("click", () => triggerResume("恢复"));

    const refresh = async () => {
      const payload = await fetchJson(`/api/jobs/${jobId}?lines=160`);
      updateText("#job-task-name", payload.task_name);
      const statusNode = $("#job-status");
      if (statusNode) {
        statusNode.innerHTML = `<span class="status-pill ${statusClass(payload.status)}">${escapeHtml(statusText(payload.status))}</span>`;
      }
      updateText("#job-started", payload.metadata?.started_at || "暂无");
      updateText("#job-completed", payload.metadata?.completed_at || "暂无");
      updateText("#job-exit-code", String(payload.metadata?.exit_code ?? "暂无"));
      updateText("#job-summary-note", payload.metadata?.summary_note || "已记录执行作业。");
      updateText("#detail-task", payload.metadata?.task_name || "暂无");
      updateText("#detail-label", payload.metadata?.job_label || "未填写");
      updateText("#detail-python", payload.metadata?.python_executable || "暂无");
      updateText("#detail-command", payload.metadata?.command_display || "暂无");
      const stdout = $("#stdout-tail");
      const stderr = $("#stderr-tail");
      if (stdout) stdout.textContent = (payload.stdout_tail || []).join("\n");
      if (stderr) stderr.textContent = (payload.stderr_tail || []).join("\n");
    };

    startPolling(refresh, 2500);
  }

  function initDoctor() {
    const refresh = async () => {
      const payload = await fetchJson("/api/doctor");
      const status = $("#doctor-overall-status");
      if (status) {
        status.textContent = statusText(payload.status);
        status.className = `signal-chip ${payload.status === "ok" ? "is-good" : "is-warn"}`;
      }
      const table = $("#doctor-check-table");
      if (table) {
        table.innerHTML = (payload.checks || [])
          .map(
            (check) => `
              <tr>
                <td>${escapeHtml(check.name)}</td>
                <td><span class="status-pill ${check.ok ? "status-succeeded" : "status-failed"}">${check.ok ? "正常" : "失败"}</span></td>
                <td>${escapeHtml(check.detail)}</td>
              </tr>
            `
          )
          .join("");
      }
    };
    startPolling(refresh, 6000);
  }

  function initTradePlan() {
    const refresh = async () => {
      const payload = await fetchJson("/api/trade-plan");
      const preview = $("#trade-plan-artifact-preview");
      if (preview) preview.textContent = (payload.preview_lines || []).join("\n");
    };
    startPolling(refresh, 5000);
  }

  function initContinuousPolicy() {
    const renderPreviewRows = (rows) =>
      (rows || [])
        .map(
          (row) => `
            <tr>
              <td>${escapeHtml(row.stock || "")}</td>
              <td>${escapeHtml(row.share_action || row.execution_action || "暂无")}</td>
              <td>${escapeHtml(row.current_weight || "0")}</td>
              <td>${escapeHtml(row.target_weight || "0")}</td>
              <td>${escapeHtml(row.current_shares || "0")}</td>
              <td>${escapeHtml(row.target_shares || "0")}</td>
              <td>${escapeHtml(row.reason_summary || "见 reasoning.json")}</td>
            </tr>
          `
        )
        .join("");

    const refresh = async () => {
      const payload = await fetchJson("/api/continuous-policy");
      const train = payload.latest_train || {};
      const evaluation = payload.latest_evaluation || {};
      const continuousMetrics = evaluation.continuous_policy_metrics || {};
      const continuityMetrics = evaluation.continuity_metrics || {};
      const teacherMetrics = evaluation.teacher_oracle_metrics || {};
      const referenceMetrics = evaluation.active_manifest_reference?.metrics || {};
      const latestExport = payload.latest_export || {};
      const latestProtocol = payload.latest_protocol || {};
      const latestAudit = payload.latest_behavior_audit || {};
      const latestLedger = payload.latest_conclusion_ledger || {};
      const protocolShadow = latestProtocol.shadow || {};
      const protocolTrainWindow = latestProtocol.train_window || {};
      const protocolEvalWindow = latestProtocol.evaluation_window || {};
      const protocolShadowWindow = latestProtocol.shadow_window || {};
      const runtimeState = payload.runtime_state || {};

      updateText("#cp-latest-model", train.model_artifact_path || "暂无");
      updateText("#cp-latest-signal", latestExport.signal_date || "暂无");
      updateText("#cp-latest-protocol", latestProtocol.run_tag || "暂无");
      updateText("#cp-runtime-holdings", String(Object.keys(runtimeState.holdings || {}).length));
      updateText(
        "#cp-runtime-gap",
        latestExport.runtime_alignment_gap !== null && latestExport.runtime_alignment_gap !== undefined
          ? String(latestExport.runtime_alignment_gap)
          : "暂无"
      );

      updateText("#cp-train-tag", train.run_tag || "暂无");
      updateText("#cp-train-at", train.trained_at || "暂无");
      updateText("#cp-train-backend", train.trainer_backend || "暂无");
      updateText("#cp-train-decoder", train.decoder_profile || "暂无");
      updateText("#cp-train-contract", train.training_contract?.contract_class || "暂无");
      updateText("#cp-train-pool", train.pool_name || "暂无");
      updateText("#cp-train-preset", train.label_preset || "暂无");
      updateText("#cp-train-samples", train.sample_rows !== undefined ? String(train.sample_rows) : "暂无");
      updateText("#cp-train-features", train.feature_count !== undefined ? String(train.feature_count) : "暂无");
      updateText(
        "#cp-train-epochs",
        train.training_diagnostics?.completed_epochs !== undefined ? String(train.training_diagnostics.completed_epochs) : "暂无"
      );
      updateText(
        "#cp-train-best-epoch",
        train.training_diagnostics?.best_epoch !== undefined ? String(train.training_diagnostics.best_epoch) : "暂无"
      );
      updateText("#cp-train-path", train.model_artifact_path || "暂无");

      updateText("#cp-eval-annual", continuousMetrics.annual_return !== undefined ? String(continuousMetrics.annual_return) : "暂无");
      updateText("#cp-eval-sharpe", continuousMetrics.sharpe !== undefined ? String(continuousMetrics.sharpe) : "暂无");
      updateText("#cp-eval-mdd", continuousMetrics.max_drawdown !== undefined ? String(continuousMetrics.max_drawdown) : "暂无");
      updateText("#cp-eval-turnover", continuousMetrics.avg_turnover !== undefined ? String(continuousMetrics.avg_turnover) : "暂无");
      updateText("#cp-teacher-annual", teacherMetrics.annual_return !== undefined ? String(teacherMetrics.annual_return) : "暂无");
      updateText("#cp-reference-annual", referenceMetrics.annual_return !== undefined ? String(referenceMetrics.annual_return) : "暂无");
      updateText("#cp-cont-add-win", continuityMetrics.add_win_rate_5d !== undefined ? String(continuityMetrics.add_win_rate_5d) : "暂无");
      updateText(
        "#cp-cont-reduce",
        continuityMetrics.reduce_success_rate_5d !== undefined ? String(continuityMetrics.reduce_success_rate_5d) : "暂无"
      );
      updateText(
        "#cp-cont-exit",
        continuityMetrics.exit_timeliness_rate_5d !== undefined ? String(continuityMetrics.exit_timeliness_rate_5d) : "暂无"
      );
      updateText(
        "#cp-cont-cash",
        continuityMetrics.cash_timing_quality_1d !== undefined ? String(continuityMetrics.cash_timing_quality_1d) : "暂无"
      );
      updateText(
        "#cp-cont-hold",
        continuityMetrics.avg_hold_days_before_action !== undefined ? String(continuityMetrics.avg_hold_days_before_action) : "暂无"
      );
      updateText(
        "#cp-cont-reentry",
        continuityMetrics.reentry_within_10d_rate !== undefined ? String(continuityMetrics.reentry_within_10d_rate) : "暂无"
      );
      updateText(
        "#cp-cont-hold-share",
        continuityMetrics.hold_share !== undefined ? String(continuityMetrics.hold_share) : "暂无"
      );
      updateText(
        "#cp-cont-hold-quality",
        continuityMetrics.hold_retention_quality_5d !== undefined ? String(continuityMetrics.hold_retention_quality_5d) : "暂无"
      );
      updateText(
        "#cp-cont-reduce-quality",
        continuityMetrics.reduce_preservation_quality_5d !== undefined ? String(continuityMetrics.reduce_preservation_quality_5d) : "暂无"
      );
      updateText(
        "#cp-cont-reversal",
        continuityMetrics.immediate_reversal_rate_3d !== undefined ? String(continuityMetrics.immediate_reversal_rate_3d) : "暂无"
      );
      updateText("#cp-protocol-tag", latestProtocol.run_tag || "暂无");
      updateText("#cp-protocol-backend", latestProtocol.trainer_backend || train.trainer_backend || "暂无");
      updateText("#cp-protocol-decoder", latestProtocol.decoder_profile || train.decoder_profile || "暂无");
      updateText("#cp-protocol-contract", latestProtocol.training_contract?.contract_class || train.training_contract?.contract_class || "暂无");
      updateText(
        "#cp-protocol-train-window",
        protocolTrainWindow.start_date && protocolTrainWindow.end_date
          ? `${protocolTrainWindow.start_date} -> ${protocolTrainWindow.end_date}`
          : "暂无"
      );
      updateText(
        "#cp-protocol-eval-window",
        protocolEvalWindow.start_date && protocolEvalWindow.end_date
          ? `${protocolEvalWindow.start_date} -> ${protocolEvalWindow.end_date}`
          : "暂无"
      );
      updateText(
        "#cp-protocol-shadow-window",
        protocolShadowWindow.start_date && protocolShadowWindow.end_date
          ? `${protocolShadowWindow.start_date} -> ${protocolShadowWindow.end_date}`
          : "暂无"
      );
      updateText("#cp-protocol-preset", latestProtocol.label_preset || train.label_preset || "暂无");
      updateText(
        "#cp-protocol-shadow-annual",
        protocolShadow.metrics?.annual_return !== undefined ? String(protocolShadow.metrics.annual_return) : "暂无"
      );
      updateText(
        "#cp-protocol-promotable",
        latestProtocol.training_contract?.promotable ? "是" : latestProtocol.training_contract ? "否" : "暂无"
      );
      updateText(
        "#cp-protocol-gate",
        latestProtocol.promotion_gate?.status ? String(latestProtocol.promotion_gate.status) : "暂无"
      );
      updateText("#cp-protocol-summary-json", latestProtocol.protocol_summary_json || protocolShadow.shadow_summary_json || "暂无");
      updateText("#cp-audit-path", latestAudit.output_path || "暂无");
      updateText("#cp-audit-top", latestAudit.bottlenecks?.[0]?.name || "暂无");
      updateText("#cp-ledger-path", latestLedger.output_path || "暂无");
      updateText(
        "#cp-ledger-stage-count",
        Array.isArray(latestLedger.stage_local_conclusions) ? String(latestLedger.stage_local_conclusions.length) : "0"
      );

      updateText("#cp-export-date", latestExport.signal_date || "暂无");
      updateText("#cp-export-dir", latestExport.export_dir || "暂无");
      updateText("#cp-export-csv", latestExport.action_panel_csv || "暂无");
      updateText("#cp-export-reasoning", latestExport.reasoning_json || "暂无");
      updateText("#cp-export-runtime", latestExport.runtime_state_json || "暂无");
      updateText(
        "#cp-export-gap",
        latestExport.runtime_alignment_gap !== null && latestExport.runtime_alignment_gap !== undefined
          ? String(latestExport.runtime_alignment_gap)
          : "暂无"
      );

      updateText("#cp-runtime-date", runtimeState.last_signal_date || "暂无");
      updateText(
        "#cp-runtime-cash",
        runtimeState.cash_weight !== null && runtimeState.cash_weight !== undefined ? String(runtimeState.cash_weight) : "暂无"
      );
      updateText("#cp-runtime-count", String(Object.keys(runtimeState.holdings || {}).length));
      updateText("#cp-runtime-model", runtimeState.latest_model_artifact || "暂无");
      updateText("#cp-runtime-export-dir", runtimeState.latest_export_dir || "暂无");
      updateText(
        "#cp-runtime-gap-detail",
        runtimeState.runtime_alignment_gap !== null && runtimeState.runtime_alignment_gap !== undefined
          ? String(runtimeState.runtime_alignment_gap)
          : "暂无"
      );

      const tableBody = $("#cp-action-table-body");
      if (tableBody) tableBody.innerHTML = renderPreviewRows(payload.action_panel_preview || []);
    };

    startPolling(refresh, 4500);
  }

  function buildAccountRow(position) {
    const row = position || {};
    return `
      <tr class="account-position-row">
        <td><input type="text" class="account-stock-input" value="${escapeHtml(row.stock || "")}" placeholder="600000.SH"></td>
        <td><input type="number" class="account-shares-input" min="1" step="1" value="${escapeHtml(row.shares ?? "")}" placeholder="1000"></td>
        <td><input type="number" class="account-cost-input" min="0" step="0.01" value="${escapeHtml(row.cost_price ?? "")}" placeholder="10.52"></td>
        <td><button class="inline-button inline-button--danger" type="button" data-action="remove-position">删除</button></td>
      </tr>
    `;
  }

  function showAccountResult(message, isError) {
    const result = $("#account-action-result");
    if (!result) return;
    result.innerHTML = `<p>${escapeHtml(message)}</p>`;
    result.style.borderColor = isError ? "rgba(181, 58, 47, 0.35)" : "rgba(15, 123, 99, 0.35)";
  }

  function renderAccountSnapshot(payload) {
    updateText("#account-source", payload.source || "unknown");
    updateText("#account-position-count", String(payload.position_count ?? 0));
    updateText(
      "#account-cash-summary",
      payload.available_cash !== null && payload.available_cash !== undefined ? String(payload.available_cash) : "暂无"
    );
    updateText("#account-last-modified", payload.last_modified_at || "暂无");
    updateText("#account-file-path", payload.path || "暂无");
    updateText("#account-headers-status", payload.headers_ok ? "正常" : "异常");
    updateText("#account-tickers", (payload.tickers || []).join(", ") || "暂无");
    updateText("#account-total-shares", String(payload.total_shares ?? 0));
    const cashInput = $("#account-cash-input");
    if (cashInput) {
      cashInput.value = payload.available_cash !== null && payload.available_cash !== undefined ? String(payload.available_cash) : "";
    }
    const body = $("#account-positions-body");
    if (body) {
      const positions = payload.positions || [];
      body.innerHTML = positions.length ? positions.map((position) => buildAccountRow(position)).join("") : buildAccountRow({});
    }
  }

  function collectAccountPayload() {
    const cashInput = $("#account-cash-input");
    const rawCash = cashInput ? String(cashInput.value || "").trim() : "";
    const positions = Array.from(document.querySelectorAll("#account-positions-body .account-position-row"))
      .map((row) => ({
        stock: row.querySelector(".account-stock-input")?.value || "",
        shares: row.querySelector(".account-shares-input")?.value || "",
        cost_price: row.querySelector(".account-cost-input")?.value || "",
      }))
      .filter((item) => String(item.stock || "").trim() || String(item.shares || "").trim() || String(item.cost_price || "").trim());
    return {
      available_cash: rawCash === "" ? null : rawCash,
      positions,
    };
  }

  function initAccount() {
    const body = $("#account-positions-body");
    const addButton = $("#account-add-row-button");
    const saveButton = $("#account-save-button");
    const resetButton = $("#account-reset-button");
    if (!body) return;

    async function refresh() {
      const payload = await fetchJson("/api/account");
      renderAccountSnapshot(payload);
    }

    body.addEventListener("click", (event) => {
      const target = event.target;
      if (!(target instanceof HTMLElement)) return;
      if (target.dataset.action !== "remove-position") return;
      const rows = Array.from(body.querySelectorAll(".account-position-row"));
      if (rows.length === 1) {
        rows[0].remove();
        body.innerHTML = buildAccountRow({});
        return;
      }
      target.closest(".account-position-row")?.remove();
    });

    if (addButton) {
      addButton.addEventListener("click", () => {
        body.insertAdjacentHTML("beforeend", buildAccountRow({}));
      });
    }

    if (saveButton) {
      saveButton.addEventListener("click", async () => {
        showAccountResult("正在保存模拟账户...", false);
        try {
          const payload = await fetchJson("/api/account", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(collectAccountPayload()),
          });
          renderAccountSnapshot(payload);
          showAccountResult("模拟账户已保存。", false);
        } catch (error) {
          showAccountResult(error.message || "保存模拟账户失败。", true);
        }
      });
    }

    if (resetButton) {
      resetButton.addEventListener("click", async () => {
        const confirmed = window.confirm("是否按示例文件重置当前模拟账户？这会覆盖当前 current_positions.csv。");
        if (!confirmed) return;
        showAccountResult("正在按示例文件重置模拟账户...", false);
        try {
          const payload = await fetchJson("/api/account/reset-example", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
          });
          renderAccountSnapshot(payload);
          showAccountResult("模拟账户已按示例文件重置。", false);
        } catch (error) {
          showAccountResult(error.message || "重置模拟账户失败。", true);
        }
      });
    }

    refresh();
  }

  function initRuntime() {
    const unlockButton = $("#runtime-force-unlock-button");
    const result = $("#runtime-action-result");

    async function triggerUnlock() {
      const confirmed = window.confirm("是否强制清除执行运行时锁？只有在确认记录中的作业已经不再运行时才这样做。");
      if (!confirmed) return;
      if (result) {
        result.innerHTML = "<p>正在清理失效锁...</p>";
        result.style.borderColor = "rgba(173, 107, 31, 0.35)";
      }
      try {
        const payload = await fetchJson("/api/unlock", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ force: true }),
        });
        if (result) {
          result.innerHTML = `<p>${escapeHtml(payload.detail || "锁已清理。")}</p>`;
          result.style.borderColor = "rgba(15, 123, 99, 0.35)";
        }
        await refresh();
      } catch (error) {
        if (result) {
          result.innerHTML = `<p>${escapeHtml(error.message || "清理锁失败。")}</p>`;
          result.style.borderColor = "rgba(181, 58, 47, 0.35)";
        }
      }
    }

    if (unlockButton) unlockButton.addEventListener("click", triggerUnlock);

    const refresh = async () => {
      const payload = await fetchJson("/api/status?history_limit=20");
      updateText("#runtime-lock-job", payload.lock?.job_id || "未锁定");
      updateText("#runtime-active-threads", String((payload.active_thread_job_ids || []).length));
      updateText("#runtime-warning-count", String((payload.warnings || []).length));
      updateText("#runtime-updated", payload.updated_at || "暂无");
      updateText(
        "#runtime-summary-note",
        payload.lock
          ? `当前锁属于 ${payload.lock.job_id || "未知作业"}；只有在确认该作业已结束后才能强制解锁。`
          : "运行时当前未锁定；强制解锁只应用于 stale lock 恢复场景。"
      );
      const warnings = $("#runtime-warning-list");
      if (warnings) {
        warnings.innerHTML = (payload.warnings || []).length
          ? (payload.warnings || [])
              .map((warning) => `<li class="is-bad"><span>${escapeHtml(warning)}</span></li>`)
              .join("")
          : '<li class="is-good"><span>当前没有记录到运行时告警。</span></li>';
      }
      const pane = $("#runtime-json-pane");
      if (pane) pane.textContent = JSON.stringify(payload, null, 2);
    };
    startPolling(refresh, 5000);
  }

  document.addEventListener("DOMContentLoaded", () => {
    const page = document.querySelector(".shell-content")?.dataset.page;
    if (page === "dashboard") initDashboard();
    if (page === "tasks") initTasks();
    if (page === "jobs") initJobs();
    if (page === "job-detail") initJobDetail();
    if (page === "doctor") initDoctor();
    if (page === "trade-plan") initTradePlan();
    if (page === "continuous-policy") initContinuousPolicy();
    if (page === "account") initAccount();
    if (page === "runtime") initRuntime();
  });
})();
