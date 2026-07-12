from __future__ import annotations

import json
from pathlib import Path

import pytest

from daily_research.path_policy import seq100_research_generation as generation


def _contract_payload() -> dict[str, object]:
    return json.loads(
        generation.DEFAULT_DEVELOPMENT_KPI_PORTFOLIO_CONTRACT.read_text(encoding="utf-8")
    )


def test_init_development_cli_uses_candidate_complete_defaults() -> None:
    args = generation._parser().parse_args(["init-development"])

    assert Path(args.source_view) == generation.DEFAULT_DEVELOPMENT_SOURCE_VIEW
    assert "candidate_complete" in str(args.source_view)
    assert Path(args.kpi_portfolio_contract) == (
        generation.DEFAULT_DEVELOPMENT_KPI_PORTFOLIO_CONTRACT
    )


def test_kpi_portfolio_contract_validation_returns_bound_identity() -> None:
    binding = generation._validated_development_kpi_portfolio_contract(
        generation.DEFAULT_DEVELOPMENT_KPI_PORTFOLIO_CONTRACT
    )

    assert binding["contract_id"] == generation.DEVELOPMENT_KPI_PORTFOLIO_CONTRACT_ID
    assert binding["research_contract_id"] == generation.approved_development_contract_binding()[
        "contract_id"
    ]
    assert binding["top_k"] == [1, 3, 5, 10]
    assert binding["daily_cohort_cash_cny"] == 1_000_000.0
    assert binding["candidate_selection"] == {
        "execution_return_and_plan_coverage_required": 1.0,
        "label_dependent_value_coverage_must_be_reported": True,
        "must_not_be_imputed": True,
    }
    assert binding["sha256"] == generation._file_sha256(
        generation.DEFAULT_DEVELOPMENT_KPI_PORTFOLIO_CONTRACT
    )


@pytest.mark.parametrize(
    ("field_path", "replacement", "message"),
    [
        (("contract_id",), "changed", "contract id"),
        (("capital_semantics", "allocation"), "changed", "capital semantics"),
        (("execution_scenarios", "stress"), "changed", "execution scenarios"),
        (("candidate_selection", "primary"), "changed", "candidate-completeness"),
        (
            ("candidate_selection", "execution_return_and_plan_coverage_required"),
            0.99,
            "candidate-completeness",
        ),
        (
            ("candidate_selection", "label_dependent_value_coverage_must_be_reported"),
            False,
            "candidate-completeness",
        ),
        (
            ("candidate_selection", "must_not_be_imputed"),
            False,
            "candidate-completeness",
        ),
        (("protected_boundaries", "changes_qdp_active"), True, "protected boundaries"),
    ],
)
def test_kpi_portfolio_contract_rejects_semantic_drift(
    tmp_path: Path,
    field_path: tuple[str, ...],
    replacement: object,
    message: str,
) -> None:
    payload = _contract_payload()
    target: dict[str, object] = payload
    for field in field_path[:-1]:
        nested = target[field]
        assert isinstance(nested, dict)
        target = nested
    target[field_path[-1]] = replacement
    contract_path = tmp_path / "tampered_kpi_contract.json"
    contract_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        generation._validated_development_kpi_portfolio_contract(contract_path)


def test_development_registry_binds_candidate_execution_and_kpi_provenance(
    tmp_path: Path,
) -> None:
    source = tmp_path / "synthetic_source.json"
    source.write_text("{}", encoding="utf-8")

    registry = generation.initialize_development_registry(
        root=tmp_path / "study",
        source_view=source,
        store_root=tmp_path / "folds",
        profiles=("baseline",),
        require_corrected_source=False,
        require_existing_folds=False,
        evidence_policy=generation.EVIDENCE_POLICY_SYNTHETIC_ALLOWED,
    )

    expected_kpi = generation._validated_development_kpi_portfolio_contract(
        generation.DEFAULT_DEVELOPMENT_KPI_PORTFOLIO_CONTRACT
    )
    assert registry["kpi_portfolio_contract"] == expected_kpi
    provenance = {
        Path(item["path"]).resolve(): item["sha256"]
        for item in registry["code_provenance"]["files"]
    }
    candidate_execution_path = Path(
        "daily_research/path_policy/seq100_candidate_execution.py"
    ).resolve()
    kpi_path = generation.DEFAULT_DEVELOPMENT_KPI_PORTFOLIO_CONTRACT.resolve()
    assert provenance[candidate_execution_path] == generation._file_sha256(
        candidate_execution_path
    )
    assert provenance[kpi_path] == expected_kpi["sha256"]
    assert str(kpi_path) in registry["additional_provenance_paths"]
    assert all(
        job["expected_resolved_training_config"]["minimum_complete_epochs"] == 1
        for job in registry["jobs"]
    )


def test_development_aggregation_reports_partial_value_coverage_without_imputation() -> None:
    metrics = {name: 0.1 for name in generation.DEVELOPMENT_METRIC_NAMES}
    for top_k in generation.DEVELOPMENT_TOP_K:
        for metric_name in (
            "selected_realized_plan_return_coverage",
            "universe_realized_plan_return_coverage",
            "selected_realized_plan_coverage",
            "universe_realized_plan_coverage",
        ):
            metrics[f"top{top_k}_{metric_name}"] = 1.0
        metrics[f"top{top_k}_selected_realized_plan_value_coverage"] = 0.75
        metrics[f"top{top_k}_universe_realized_plan_value_coverage"] = 0.90
    records = [
        {
            "profile": "baseline",
            "development_year": year,
            "seed": generation.DEVELOPMENT_SEED,
            "metrics": metrics,
        }
        for year in generation.DEVELOPMENT_YEARS
    ]

    aggregate = generation._aggregate_development_records(records)["baseline"]

    assert aggregate["mean_topk_selected_realized_plan_value_coverage"] == 0.75
    assert aggregate["mean_topk_universe_realized_plan_value_coverage"] == 0.90
    assert aggregate["top1_net_realized_plan_value_base_alpha"] == 0.1
