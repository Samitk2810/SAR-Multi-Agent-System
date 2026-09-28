"""
State schema for the SAR generation workflow:

XGBoost -> Investigator -> AML Auditor -[escalate]-> {Regulatory Analyst, Evidence Analyst}
                                |                                |
                             [dismiss]                    (join) v
                                |                          SAR Drafter -> Compliance Auditor -[PASS]-> END
                                v                                              |
                               END                                          [FAIL, capped]
                                                                                |
                                                                          SAR Revision -> back to Compliance Auditor
"""

from __future__ import annotations
from typing import TypedDict, Literal, Annotated, Optional
from datetime import datetime, timezone
import operator


# ---------------------------------------------------------------------------
# Input: this is what your XGBoost notebook already computes per flagged row.
# No DB lookup needed -- it's the alert payload itself.
# ---------------------------------------------------------------------------
class TransactionAlert(TypedDict):
    transaction_id: str
    sender_account: str
    receiver_account: str
    amount: float
    payment_type: str
    payment_currency: str
    received_currency: str
    sender_bank_location: str
    receiver_bank_location: str
    transaction_date: str
    transaction_time: str

    # model output
    xgb_score: float
    xgb_top_features: list[str]

    # engineered features, straight from the notebook
    sender_txn_count: int
    receiver_txn_count: int
    amount_dev_from_sender_avg: float
    is_cross_border: bool
    is_round_amount: bool
    near_reporting_threshold: bool


# ---------------------------------------------------------------------------
# Sequential stage outputs
# ---------------------------------------------------------------------------
class InvestigatorBrief(TypedDict):
    anomaly_summary: str
    flagged_signals: list[str]
    initial_risk_level: Literal["low", "medium", "high"]


class AMLAuditorDecision(TypedDict):
    decision: Literal["escalate", "dismiss"]
    matched_typology: Optional[str]   # e.g. "structuring", "layering"; None if dismissed
    confidence: float                 # 0-1
    reasoning: str


# ---------------------------------------------------------------------------
# Parallel branch outputs (different keys -> no merge conflict on join)
# ---------------------------------------------------------------------------
class RegulatoryFindings(TypedDict):
    threshold_triggered: bool
    sanctions_hit: bool
    pep_hit: bool
    jurisdiction_risk: Literal["low", "medium", "high"]
    notes: str


class EvidenceSummary(TypedDict):
    consolidated_evidence: list[str]
    evidence_sources: dict[str, str]   # claim -> which field/agent it came from


# ---------------------------------------------------------------------------
# Drafting / validation loop
# ---------------------------------------------------------------------------
class SARDraft(TypedDict):
    narrative: str
    structured_fields: dict[str, str]  # FinCEN-style field -> value
    draft_version: int


class ComplianceCheck(TypedDict):
    result: Literal["pass", "fail"]
    issues: list[str]                  # empty when result == "pass"


# ---------------------------------------------------------------------------
# Trace: one entry per agent handoff, so the full decision path can be
# shown to the user (or a compliance reviewer) after the run finishes.
# ---------------------------------------------------------------------------
class TraceEntry(TypedDict):
    agent: str                 # e.g. "AML Auditor"
    action: str                # short label, e.g. "escalated to Regulatory + Evidence"
    input_summary: str         # what this agent was handed
    output_summary: str        # what it decided / produced
    metadata: dict             # anything extra: tool calls made, scores, flags
    timestamp: str             # ISO 8601, set automatically
    duration_ms: Optional[float]


def new_trace_entry(
    agent: str,
    action: str,
    input_summary: str = "",
    output_summary: str = "",
    metadata: Optional[dict] = None,
    duration_ms: Optional[float] = None,
) -> TraceEntry:
    """Build one trace entry with a consistent shape and timestamp.

    Usage inside a node:

        def aml_auditor(state: SARWorkflowState) -> dict:
            start = time.perf_counter()
            ...decision logic...
            entry = new_trace_entry(
                agent="AML Auditor",
                action=f"decision={decision['decision']}",
                input_summary=state["investigator_brief"]["anomaly_summary"],
                output_summary=decision["reasoning"],
                metadata={"matched_typology": decision["matched_typology"]},
                duration_ms=(time.perf_counter() - start) * 1000,
            )
            return {"aml_decision": decision, "trace": [entry]}

    Returning a single-item list is intentional -- the state's `trace` field
    uses operator.add as its reducer, so LangGraph appends this entry to the
    running list rather than overwriting it.
    """
    return TraceEntry(
        agent=agent,
        action=action,
        input_summary=input_summary,
        output_summary=output_summary,
        metadata=metadata or {},
        timestamp=datetime.now(timezone.utc).isoformat(),
        duration_ms=duration_ms,
    )


# ---------------------------------------------------------------------------
# Full graph state
# ---------------------------------------------------------------------------
class SARWorkflowState(TypedDict):
    # input
    alert: TransactionAlert

    # sequential
    investigator_brief: Optional[InvestigatorBrief]
    aml_decision: Optional[AMLAuditorDecision]

    # parallel branch outputs
    regulatory_findings: Optional[RegulatoryFindings]
    evidence_summary: Optional[EvidenceSummary]

    # drafting / validation loop
    sar_draft: Optional[SARDraft]
    compliance_check: Optional[ComplianceCheck]

    # reducer: each revision node returns {"revision_count": 1}, LangGraph sums it.
    # Use this in a conditional edge to cap the FAIL -> Revision loop, e.g.:
    #   if state["revision_count"] >= MAX_REVISIONS: route to "needs_human_review"
    revision_count: Annotated[int, operator.add]

    # full handoff history -- each node appends one TraceEntry via new_trace_entry().
    # This is what you'd render to the user as "what each agent did".
    trace: Annotated[list[TraceEntry], operator.add]

    # terminal status, set by whichever node ends the run
    status: Literal[
        "in_progress",
        "dismissed",
        "filed",
        "needs_human_review",
        "max_revisions_exceeded",
    ]
