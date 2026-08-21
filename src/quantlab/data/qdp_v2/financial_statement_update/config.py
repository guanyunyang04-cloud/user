"""Financial Statement Update: config responsibilities."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from quantlab.data.domains.contracts import DataDomain

UPDATE_ID = "financial_statement_quarterly_pit_v2"


LEGACY_UPDATE_ID = "financial_statement_quarterly_pit_v1"


BALANCE_CONFLICT_REPAIR_ID = "balance_extension_conflict_semantics_repair_v2"


SOURCE_SCHEMA_VERSION = 5


START_DATE = "2010-01-01"


END_DATE = "2025-12-31"


MAX_REPORT_PERIOD = "2025-09-30"


@dataclass(frozen=True)
class StatementSpec:
    name: str
    api_name: str
    domain: str
    fields: tuple[str, ...]
    metric_map: Mapping[str, str]


COMMON_FIELDS = (
    "ts_code",
    "ann_date",
    "f_ann_date",
    "end_date",
    "report_type",
    "comp_type",
    "end_type",
)


INCOME_METRICS = {
    "basic_eps": "basic_eps",
    "diluted_eps": "diluted_eps",
    "total_revenue": "total_revenue",
    "revenue": "revenue",
    "total_cogs": "total_cost",
    "oper_cost": "operating_cost",
    "sell_exp": "selling_expense",
    "admin_exp": "administrative_expense",
    "fin_exp": "finance_expense",
    "rd_exp": "research_development_expense",
    "operate_profit": "operating_profit",
    "total_profit": "total_profit",
    "income_tax": "income_tax",
    "n_income": "net_income",
    "n_income_attr_p": "parent_net_income",
    "minority_gain": "minority_income",
    "ebit": "ebit",
    "ebitda": "ebitda",
    "continued_net_profit": "continuing_net_income",
    "end_net_profit": "discontinued_net_income",
}


BALANCE_METRICS = {
    "total_share": "total_shares",
    "cap_rese": "capital_reserve",
    "undistr_porfit": "retained_earnings",
    "surplus_rese": "surplus_reserve",
    "money_cap": "cash_and_equivalents",
    "notes_receiv": "notes_receivable",
    "accounts_receiv": "accounts_receivable",
    "accounts_receiv_bill": "accounts_receivable_and_notes_reported",
    "receiv_financing": "receivables_financing",
    "oth_receiv": "other_receivables",
    "oth_rcv_total": "other_receivables_total",
    "prepayment": "prepayments",
    "inventories": "inventory",
    "total_cur_assets": "current_assets",
    "fix_assets": "fixed_assets",
    "fix_assets_total": "fixed_assets_total_reported",
    "cip": "construction_in_progress",
    "cip_total": "construction_in_progress_total_reported",
    "intan_assets": "intangible_assets",
    "goodwill": "goodwill",
    "total_nca": "noncurrent_assets",
    "total_assets": "total_assets",
    "st_borr": "short_term_borrowings",
    "notes_payable": "notes_payable",
    "acct_payable": "accounts_payable",
    "accounts_pay": "accounts_payable_and_notes_reported",
    "adv_receipts": "advances_from_customers",
    "contract_liab": "contract_liabilities",
    "payroll_payable": "employee_compensation_payable",
    "taxes_payable": "taxes_payable",
    "oth_payable": "other_payables",
    "oth_pay_total": "other_payables_total",
    "total_cur_liab": "current_liabilities",
    "lt_borr": "long_term_borrowings",
    "bond_payable": "bonds_payable",
    "total_ncl": "noncurrent_liabilities",
    "total_liab": "total_liabilities",
    "minority_int": "minority_equity",
    "total_hldr_eqy_exc_min_int": "parent_equity",
    "total_hldr_eqy_inc_min_int": "total_equity",
    "total_liab_hldr_eqy": "liabilities_and_equity",
}


CASH_FLOW_METRICS = {
    "net_profit": "net_profit",
    "c_fr_sale_sg": "cash_received_from_sales",
    "recp_tax_rends": "tax_refunds_received",
    "c_paid_goods_s": "cash_paid_for_goods",
    "c_paid_to_for_empl": "payroll_paid",
    "c_paid_for_taxes": "taxes_paid",
    "oth_cash_pay_oper_act": "other_operating_cash_paid",
    "st_cash_out_act": "operating_cash_outflow",
    "n_cashflow_act": "net_operating_cash_flow",
    "stot_inflows_inv_act": "investing_cash_inflow",
    "stot_out_inv_act": "investing_cash_outflow",
    "n_cashflow_inv_act": "net_investing_cash_flow",
    "c_recp_borrow": "borrowings_received",
    "proc_issue_bonds": "bond_proceeds",
    "stot_cash_in_fnc_act": "financing_cash_inflow",
    "c_prepay_amt_borr": "borrowings_repaid",
    "c_pay_dist_dpcp_int_exp": "dividends_and_interest_paid",
    "stot_cashout_fnc_act": "financing_cash_outflow",
    "n_cash_flows_fnc_act": "net_financing_cash_flow",
    "free_cashflow": "free_cash_flow",
    "n_incr_cash_cash_equ": "net_cash_increase",
    "c_cash_equ_beg_period": "cash_begin",
    "c_cash_equ_end_period": "cash_end",
}


STATEMENT_SPECS = (
    StatementSpec(
        name="income",
        api_name="income_vip",
        domain=DataDomain.INCOME_STATEMENT_QUARTERLY,
        fields=COMMON_FIELDS + tuple(INCOME_METRICS) + ("update_flag",),
        metric_map=INCOME_METRICS,
    ),
    StatementSpec(
        name="balance_sheet",
        api_name="balancesheet_vip",
        domain=DataDomain.BALANCE_SHEET_QUARTERLY,
        fields=COMMON_FIELDS + tuple(BALANCE_METRICS) + ("update_flag",),
        metric_map=BALANCE_METRICS,
    ),
    StatementSpec(
        name="cash_flow",
        api_name="cashflow_vip",
        domain=DataDomain.CASH_FLOW_STATEMENT_QUARTERLY,
        fields=COMMON_FIELDS + tuple(CASH_FLOW_METRICS) + ("update_flag",),
        metric_map=CASH_FLOW_METRICS,
    ),
)


V2_STATEMENT_SPECS = tuple(spec for spec in STATEMENT_SPECS if spec.domain == DataDomain.BALANCE_SHEET_QUARTERLY)


BALANCE_DERIVED_COLUMNS = (
    "customer_advances_and_contract_liabilities",
    "customer_liability_field_state",
    "other_receivables_total_field_state",
    "other_payables_total_field_state",
    "contract_liabilities_field_state",
    "trade_receivables_total",
    "trade_receivables_to_current_assets",
    "trade_receivables_to_total_assets",
    "trade_receivables_field_state",
    "fixed_assets_measure",
    "fixed_assets_measure_field_state",
    "construction_in_progress_measure",
    "construction_in_progress_measure_field_state",
    "trade_payables_total",
    "trade_payables_to_current_liabilities",
    "trade_payables_to_total_assets",
    "trade_payables_field_state",
)


BALANCE_EXTENSION_NUMERIC_COLUMNS = (
    "other_receivables_total",
    "other_payables_total",
    "contract_liabilities",
    "customer_advances_and_contract_liabilities",
)


BALANCE_EXTENSION_CONFLICT_COLUMN = "balance_extension_source_conflict"


BALANCE_SEMANTIC_SOURCE_COLUMNS = (
    "accounts_receivable_and_notes_reported",
    "receivables_financing",
    "fixed_assets_total_reported",
    "construction_in_progress_total_reported",
    "accounts_payable_and_notes_reported",
)


BALANCE_SEMANTIC_NUMERIC_COLUMNS = (
    "trade_receivables_total",
    "trade_receivables_to_current_assets",
    "trade_receivables_to_total_assets",
    "fixed_assets_measure",
    "construction_in_progress_measure",
    "trade_payables_total",
    "trade_payables_to_current_liabilities",
    "trade_payables_to_total_assets",
)


BALANCE_SEMANTIC_STATE_COLUMNS = (
    "trade_receivables_field_state",
    "fixed_assets_measure_field_state",
    "construction_in_progress_measure_field_state",
    "trade_payables_field_state",
)


BALANCE_SEMANTIC_CONFLICT_COLUMN = "balance_semantic_source_conflict"


COMMON_OUTPUT_COLUMNS = (
    "symbol",
    "trade_date",
    "source_date",
    "feature_available_date",
    "report_date",
    "fiscal_year",
    "fiscal_quarter",
    "announcement_date",
    "actual_announcement_date",
    "publish_date",
    "report_type",
    "company_type",
    "period_type",
    "update_flag",
)


TRAILING_OUTPUT_COLUMNS = (
    "source_duplicate_count",
    "source_conflict",
    "lag_policy",
    "source",
)


PRIMARY_KEY = (
    "publish_date",
    "symbol",
    "report_date",
    "report_type",
    "company_type",
    "period_type",
)


class FinancialStatementUpdateError(RuntimeError):
    pass
