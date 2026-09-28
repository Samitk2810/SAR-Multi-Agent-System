"""
Investigator agent: first node in the SAR workflow.

Takes the raw XGBoost alert (already contains engineered features, no DB needed)
and produces a structured InvestigatorBrief: what's anomalous about this transaction
and why, in plain analytic terms, ready for the AML Auditor to make a typology call.
"""

from __future__ import annotations
import time
import json
import os
from dotenv import load_dotenv
load_dotenv()
 

from langchain_openrouter import ChatOpenRouter

from graph.sar_workflow_state import (
    SARWorkflowState,
    TransactionAlert,
    InvestigatorBrief,
    new_trace_entry,
)

from pydantic import BaseModel
from typing import Literal


class _InvestigatorBriefSchema(BaseModel):
    anomaly_summary: str
    flagged_signals: list[str]
    initial_risk_level: Literal["low", "medium", "high"]


SYSTEM_PROMPT = """You are a financial crimes Investigator analyzing a transaction \
that has been flagged by an automated ML monitoring system.

Your job is NOT to decide whether this is money laundering -- that comes later. \
Your job is to write a clear, factual analytic summary of what is anomalous about \
this transaction, grounded ONLY in the data provided. Do not speculate about intent \
or invent facts not present in the alert.

Consider:
- Why might the model have flagged this (use the model score and top features as a hint)
- Which engineered signals stand out (cross-border, round amount, deviation from the
  sender's own typical behavior, proximity to reporting thresholds, transaction frequency)
- Whether multiple signals reinforce each other, or whether this looks like a single
  weak signal that might not warrant escalation

Assign an initial_risk_level of "low", "medium", or "high" based on how many signals
line up and how strong they are -- this is a preliminary read, not a final decision."""


def _format_alert_for_prompt(alert: TransactionAlert) -> str:
    return json.dumps(alert, indent=2, default=str)


def build_investigator_llm():
    llm = ChatOpenRouter(
    model="liquid/lfm-2.5-2.6b:free",
    temperature=0,
)
    return llm.with_structured_output(_InvestigatorBriefSchema)


_investigator_llm = None


def investigator_node(state: SARWorkflowState) -> dict:
    """LangGraph node. Reads state['alert'], writes state['investigator_brief'] + trace."""
    global _investigator_llm
    if _investigator_llm is None:
        _investigator_llm = build_investigator_llm()

    start = time.perf_counter()
    alert = state["alert"]

    messages = [
        ("system", SYSTEM_PROMPT),
        ("human", f"Alert data:\n{_format_alert_for_prompt(alert)}"),
    ]

    brief: InvestigatorBrief = _investigator_llm.invoke(messages).model_dump()
    duration_ms = (time.perf_counter() - start) * 1000

    trace = new_trace_entry(
        agent="Investigator",
        action=f"initial_risk_level={brief['initial_risk_level']}",
        input_summary=f"transaction_id={alert['transaction_id']}, xgb_score={alert['xgb_score']:.3f}",
        output_summary=brief["anomaly_summary"],
        metadata={"flagged_signals": brief["flagged_signals"]},
        duration_ms=duration_ms,
    )

    return {"investigator_brief": brief, "trace": [trace]}


# ---------------------------------------------------------------------------
# Quick manual test -- run this file directly to sanity-check the node
# without needing the rest of the graph built yet.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    sample_alert: TransactionAlert = {
        "transaction_id": "TXN-000123",
        "sender_account": "ACC-A1",
        "receiver_account": "ACC-B2",
        "amount": 9750.00,
        "payment_type": "Wire",
        "payment_currency": "USD",
        "received_currency": "USD",
        "sender_bank_location": "USA",
        "receiver_bank_location": "Cayman Islands",
        "xgb_score": 0.87,
        "xgb_top_features": ["Near_reporting_threshold", "Is_cross_border", "Amount_dev_from_sender_avg"],
        "sender_txn_count": 42,
        "receiver_txn_count": 3,
        "amount_dev_from_sender_avg": 4.2,
        "is_cross_border": True,
        "is_round_amount": False,
        "near_reporting_threshold": True,
    }

    test_state: SARWorkflowState = {
        "alert": sample_alert,
        "investigator_brief": None,
        "aml_decision": None,
        "regulatory_findings": None,
        "evidence_summary": None,
        "sar_draft": None,
        "compliance_check": None,
        "revision_count": 0,
        "trace": [],
        "status": "in_progress",
    }

    result = investigator_node(test_state)
    print(json.dumps(result, indent=2, default=str))