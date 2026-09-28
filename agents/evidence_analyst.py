"""
Evidence Analyst agent: parallel branch B (runs alongside Regulatory Analyst
after AML Auditor escalates).

No external tools needed here -- this node's job is pure consolidation, not
investigation. It takes everything gathered so far (alert, Investigator brief,
AML Auditor decision) and turns it into a clean, traceable evidence package:
one factual claim per line, each tagged to exactly where it came from.

This traceability is what makes the Compliance Auditor's groundedness check
possible later -- SAR Drafter should only ever assert things that appear in
evidence_sources here, and the Auditor can verify that mechanically.
"""

from __future__ import annotations
import time
import json

from pydantic import BaseModel
from langchain_groq import ChatGroq

from graph.sar_workflow_state import (
    SARWorkflowState,
    EvidenceSummary,
    new_trace_entry,
)


class _EvidenceSummarySchema(BaseModel):
    consolidated_evidence: list[str]
    evidence_sources: dict[str, str]


SYSTEM_PROMPT = """You are an Evidence Analyst preparing the factual record for a \
potential Suspicious Activity Report. You are NOT drafting the report and NOT \
making the escalation decision -- that has already happened. Your only job is to \
consolidate everything established so far into a clean list of factual claims, \
each one traceable to its source.

Rules:
- Every item in consolidated_evidence must be a single, specific, factual claim
  (e.g. "Transaction amount of $9,750 is within $250 of the $10,000 CTR reporting
  threshold"), not a vague statement (e.g. "the amount looks suspicious").
- Do NOT add new claims, inferences, or interpretations beyond what the alert data,
  Investigator brief, and AML Auditor decision already established. If something
  wasn't explicitly stated upstream, it does not belong here.
- evidence_sources must map each claim (use the exact claim text as the key) to
  where it came from -- e.g. "alert.amount", "alert.near_reporting_threshold",
  "Investigator brief", "AML Auditor: matched_typology". This mapping is what lets
  a later step verify the SAR narrative doesn't contain unsupported claims.
- Keep claims atomic. Split compound observations into separate list items rather
  than bundling multiple facts into one sentence."""


def build_evidence_llm():
    llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)
    return llm.with_structured_output(_EvidenceSummarySchema)


_evidence_llm = None


def evidence_analyst_node(state: SARWorkflowState) -> dict:
    """LangGraph node. Reads alert + investigator_brief + aml_decision, writes state['evidence_summary'] + trace."""
    global _evidence_llm
    if _evidence_llm is None:
        _evidence_llm = build_evidence_llm()

    start = time.perf_counter()
    alert = state["alert"]

    human_input = json.dumps(
        {
            "alert": alert,
            "investigator_brief": state["investigator_brief"],
            "aml_decision": state["aml_decision"],
        },
        indent=2,
        default=str,
    )

    messages = [
        ("system", SYSTEM_PROMPT),
        ("human", f"Case record to consolidate:\n{human_input}"),
    ]

    summary: EvidenceSummary = _evidence_llm.invoke(messages).model_dump()
    duration_ms = (time.perf_counter() - start) * 1000

    trace = new_trace_entry(
        agent="Evidence Analyst",
        action=f"consolidated {len(summary['consolidated_evidence'])} evidence items",
        input_summary=f"transaction_id={alert['transaction_id']}",
        output_summary="; ".join(summary["consolidated_evidence"][:3])
        + ("..." if len(summary["consolidated_evidence"]) > 3 else ""),
        metadata={"evidence_count": len(summary["consolidated_evidence"])},
        duration_ms=duration_ms,
    )

    return {"evidence_summary": summary, "trace": [trace]}


# ---------------------------------------------------------------------------
# Quick manual test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    sample_alert = {
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

    sample_brief = {
        "anomaly_summary": (
            "Transaction of $9,750 sits just under the $10,000 reporting threshold, "
            "is a cross-border wire to the Cayman Islands, and deviates 4.2 standard "
            "deviations from this sender's typical transaction amount. The receiving "
            "account has very low prior transaction volume (3 total)."
        ),
        "flagged_signals": [
            "near_reporting_threshold",
            "is_cross_border",
            "high amount deviation from sender baseline",
            "low receiver account history",
        ],
        "initial_risk_level": "high",
    }

    sample_decision = {
        "decision": "escalate",
        "matched_typology": "structuring",
        "confidence": 0.82,
        "reasoning": (
            "Amount just under reporting threshold combined with cross-border transfer "
            "to a high-risk jurisdiction and a receiving account with minimal prior "
            "activity is consistent with structuring to evade CTR reporting."
        ),
    }

    test_state: SARWorkflowState = {
        "alert": sample_alert,
        "investigator_brief": sample_brief,
        "aml_decision": sample_decision,
        "regulatory_findings": None,
        "evidence_summary": None,
        "sar_draft": None,
        "compliance_check": None,
        "revision_count": 0,
        "trace": [],
        "status": "in_progress",
    }

    result = evidence_analyst_node(test_state)
    print(json.dumps(result, indent=2, default=str))