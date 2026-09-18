"""Compute headline credit ratios from a company's XBRL company-facts JSON.

We deliberately stick to us-gaap tags that are near-universally tagged
(Assets, Liabilities, StockholdersEquity, ...) rather than a financial-data
API, so every number here traces back to a specific filing.
"""
from __future__ import annotations

TAGS = {
    "current_assets": "AssetsCurrent",
    "current_liabilities": "LiabilitiesCurrent",
    "equity": "StockholdersEquity",
    "lt_debt_noncurrent": "LongTermDebtNoncurrent",
    "lt_debt_current": "LongTermDebtCurrent",
    "operating_income": "OperatingIncomeLoss",
    "interest_expense": "InterestExpense",
    "depreciation_amortization": "DepreciationDepletionAndAmortization",
    "cash": "CashAndCashEquivalentsAtCarryingValue",
    "revenues": "Revenues",
}


def _entries(facts: dict, tag: str) -> list[dict]:
    node = facts.get("facts", {}).get("us-gaap", {}).get(tag)
    if not node:
        return []
    out = []
    for entries in node.get("units", {}).values():
        out.extend(entries)
    return out


def latest_annual_value(facts: dict, tag: str) -> dict | None:
    """Most recent value tagged against a full-year 10-K (fp == FY)."""
    entries = [e for e in _entries(facts, tag) if e.get("form") == "10-K" and e.get("fp") == "FY"]
    if not entries:
        return None
    entries.sort(key=lambda e: e["end"])
    return entries[-1]


def _safe_div(a: float | None, b: float | None) -> float | None:
    if a is None or b in (None, 0):
        return None
    return a / b


def compute_ratios(facts: dict) -> dict:
    def val(key: str) -> float | None:
        entry = latest_annual_value(facts, TAGS[key])
        return entry["val"] if entry else None

    def as_of(key: str) -> str | None:
        entry = latest_annual_value(facts, TAGS[key])
        return entry["end"] if entry else None

    current_assets = val("current_assets")
    current_liabilities = val("current_liabilities")
    equity = val("equity")
    operating_income = val("operating_income")
    interest_expense = val("interest_expense")
    d_and_a = val("depreciation_amortization")

    lt_debt_noncurrent = val("lt_debt_noncurrent") or 0
    lt_debt_current = val("lt_debt_current") or 0
    lt_debt = lt_debt_noncurrent + lt_debt_current if (lt_debt_noncurrent or lt_debt_current) else None

    ebitda_approx = (
        operating_income + d_and_a
        if operating_income is not None and d_and_a is not None
        else operating_income
    )

    return {
        "as_of": as_of("equity") or as_of("current_assets"),
        "current_ratio": _safe_div(current_assets, current_liabilities),
        "debt_to_equity": _safe_div(lt_debt, equity),
        "debt_to_ebitda": _safe_div(lt_debt, ebitda_approx),
        "interest_coverage": _safe_div(operating_income, interest_expense),
        "raw_inputs": {
            "current_assets": current_assets,
            "current_liabilities": current_liabilities,
            "long_term_debt": lt_debt,
            "stockholders_equity": equity,
            "operating_income": operating_income,
            "interest_expense": interest_expense,
            "ebitda_approx": ebitda_approx,
        },
    }
