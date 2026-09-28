"""
AML Auditor agent: second node in the SAR workflow.

Reads the Investigator's brief and makes the escalate/dismiss call -- this is the
first real decision point in the graph, since XGBoost is tuned for high recall
(deliberately noisy) and most alerts reaching here should NOT all turn into SAR
drafts. Also assigns a matched typology when escalating, which downstream agents
use to focus their analysis.

No database/vector store needed -- a compact typology reference is embedded
directly in the system prompt, which is enough grounding for the LLM to reason
against without extra infrastructure.
"""

from __future__ import annotations
import time
import json

from typing import Literal, Optional
from pydantic import BaseModel
from langchain_openrouter import ChatOpenRouter

from graph.sar_workflow_state import (
    SARWorkflowState,
    InvestigatorBrief,
    AMLAuditorDecision,
    new_trace_entry,
)


class _AMLAuditorDecisionSchema(BaseModel):
    decision: Literal["escalate", "dismiss"]
    matched_typology: Optional[str] = None
    confidence: float
    reasoning: str


TYPOLOGY_REFERENCE = """
Common money laundering typologies (FATF / FinCEN):

- structuring: breaking a large sum into multiple smaller transactions, often just
  under reporting thresholds, to avoid triggering mandatory reports.
- layering: moving funds through multiple accounts, institutions, or jurisdictions
  in quick succession to obscure the origin of funds.
- smurfing: using multiple individuals ("smurfs") to make small deposits/transfers
  on behalf of one beneficial owner, avoiding scrutiny on any single actor.
- trade-based laundering: disguising proceeds via over/under-invoicing of goods or
  services in international trade.
- shell layering: routing funds through shell companies with no real business
  activity to break the audit trail.
- rapid movement / pass-through: funds arrive and leave an account almost
  immediately, with little relation to the account's normal activity pattern.
"""

SYSTEM_PROMPT = f"""You are an AML (Anti-Money Laundering) Auditor. You receive an \
Investigator's brief on a flagged transaction and must decide whether it warrants \
full escalation for SAR (Suspicious Activity Report) drafting, or dismissal as a \
false positive.

{TYPOLOGY_REFERENCE}

Decision guidance:
- ESCALATE only when the evidence plausibly matches a known typology AND multiple
  independent signals reinforce each other. A single weak signal (e.g. only
  "cross-border" with no other anomaly) is usually not enough on its own.
- DISMISS when the anomaly has an innocent, common explanation, or when signals
  are weak/contradictory. Be willing to dismiss -- the ML model is intentionally
  noisy, and drafting SARs on weak evidence wastes compliance officers' time and
  can itself create regulatory risk.
- Set matched_typology to one of the typologies above (or a close variant) when
  escalating; leave it as null/None when dismissing.
- confidence is your calibrated confidence in the decision itself (0-1), not in
  whether laundering occurred.
- reasoning must be specific to this case -- reference the actual signals, not
  generic language."""


def build_aml_auditor_llm():
    llm = ChatOpenRouter(
    model="liquid/lfm-2.5-2.6b:free",
    temperature=0,
)
    return llm.with_structured_output(_AMLAuditorDecisionSchema)


_aml_auditor_llm = None


def aml_auditor_node(state: SARWorkflowState) -> dict:
    """LangGraph node. Reads state['investigator_brief'], writes state['aml_decision'] + trace."""
    global _aml_auditor_llm
    if _aml_auditor_llm is None:
        _aml_auditor_llm = build_aml_auditor_llm()

    start = time.perf_counter()
    alert = state["alert"]
    brief: InvestigatorBrief = state["investigator_brief"]

    human_input = json.dumps(
        {
            "xgb_score": alert["xgb_score"],
            "xgb_top_features": alert["xgb_top_features"],
            "investigator_brief": brief,
        },
        indent=2,
        default=str,
    )

    messages = [
        ("system", SYSTEM_PROMPT),
        ("human", f"Investigator findings:\n{human_input}"),
    ]

    decision: AMLAuditorDecision = _aml_auditor_llm.invoke(messages).model_dump()
    duration_ms = (time.perf_counter() - start) * 1000

    trace = new_trace_entry(
        agent="AML Auditor",
        action=f"decision={decision['decision']}"
        + (f", typology={decision['matched_typology']}" if decision["decision"] == "escalate" else ""),
        input_summary=brief["anomaly_summary"],
        output_summary=decision["reasoning"],
        metadata={"confidence": decision["confidence"]},
        duration_ms=duration_ms,
    )

    status = "dismissed" if decision["decision"] == "dismiss" else "in_progress"

    return {"aml_decision": decision, "trace": [trace], "status": status}


# ---------------------------------------------------------------------------
# Quick manual test -- chains off the Investigator's output shape so you can
# test this node in isolation before the graph exists.
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

    sample_brief: InvestigatorBrief = {
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

    test_state: SARWorkflowState = {
        "alert": sample_alert,
        "investigator_brief": sample_brief,
        "aml_decision": None,
        "regulatory_findings": None,
        "evidence_summary": None,
        "sar_draft": None,
        "compliance_check": None,
        "revision_count": 0,
        "trace": [],
        "status": "in_progress",
    }

    result = aml_auditor_node(test_state)
    print(json.dumps(result, indent=2, default=str))