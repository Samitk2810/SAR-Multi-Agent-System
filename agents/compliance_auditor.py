"""
Compliance Auditor agent: validates the SAR draft before it's allowed to reach
a human for sign-off (or loop back for revision).

Two checks, deliberately split by how they should be done:

1. Field completeness -- DETERMINISTIC. Checks structured_fields against the
   same FinCEN template SAR Drafter used (imported, not redefined, so there's
   one source of truth for what FinCEN actually requires). This is a mechanical
   presence check -- no LLM judgment needed or wanted here.

2. Groundedness + narrative completeness -- LLM-assisted, because it requires
   semantic comparison (does this narrative sentence actually correspond to an
   evidence item, even if worded differently?) that a plain string match can't
   do reliably. This is the hallucination firewall: every claim in the
   narrative must trace back to evidence_summary, or the draft fails.

Both feed into one pass/fail result with a combined, specific issue list that
Revision Agent can act on directly.
"""

from __future__ import annotations
import time
import json
from typing import TypedDict

from pydantic import BaseModel
from langchain_groq import ChatGroq

from graph.sar_workflow_state import (
    SARWorkflowState,
    ComplianceCheck,
    new_trace_entry,
)
from agents.sar_drafter import get_fincen_sar_template


# ---------------------------------------------------------------------------
# Check 1: field completeness (deterministic)
# ---------------------------------------------------------------------------
def check_field_completeness(structured_fields: dict) -> list[str]:
    """Every key from Parts I-IV of the FinCEN template must be present
    (value can legitimately be "Unknown" -- that's fine; MISSING is not)."""
    template = get_fincen_sar_template()
    required_keys = [
        key
        for section in (
            "part_i_subject_information",
            "part_ii_suspicious_activity_information",
            "part_iii_financial_institution_information",
            "part_iv_filing_institution_contact_information",
        )
        for key in template[section]
    ]

    issues = []
    for key in required_keys:
        if key not in structured_fields:
            issues.append(f"Missing required field: {key}")
        elif not str(structured_fields[key]).strip():
            issues.append(f"Field present but empty: {key} (use 'Unknown', not blank)")

    return issues


# ---------------------------------------------------------------------------
# Check 2: groundedness + narrative completeness (LLM, structured output)
# ---------------------------------------------------------------------------
class GroundednessReview(TypedDict):
    fully_grounded: bool
    ungrounded_claims: list[str]          # narrative sentences not traceable to evidence
    missing_narrative_elements: list[str]  # any of who/what/when/where/why/how not addressed


class _GroundednessReviewSchema(BaseModel):
    fully_grounded: bool
    ungrounded_claims: list[str]
    missing_narrative_elements: list[str]


GROUNDEDNESS_SYSTEM_PROMPT = """You are a compliance reviewer checking a SAR \
narrative for two things ONLY:

1. GROUNDEDNESS: Does every factual claim in the narrative trace back to EITHER
   consolidated_evidence OR regulatory_findings -- both are valid, already-vetted
   sources; the drafter is allowed to use both. For example, a claim that
   jurisdiction risk is high is grounded if regulatory_findings.jurisdiction_risk
   is "high"; a claim about sanctions/PEP status is grounded if it matches
   regulatory_findings.sanctions_hit / pep_hit / notes. Only flag a sentence if
   it introduces a fact, number, date, characterization, RECOMMENDATION, or
   ACTION ITEM that is not supported by EITHER source, even if it sounds
   plausible. Paraphrasing either source is fine; adding unsupported detail,
   or stating that some action "has been" or "should be" taken when neither
   source says so, is not.

2. COMPLETENESS: Does the narrative address all six required elements -- who,
   what, when, where, why, how? If the narrative explicitly states an element
   is unknown/unavailable (e.g. "subject identity is not established"), that
   COUNTS as addressed -- an honest gap is not a completeness failure. Only
   flag an element if the narrative is silent on it entirely.

Do not comment on writing style, tone, or anything else. For ungrounded_claims,
quote the exact problematic sentence VERBATIM from the narrative (not a
paraphrase) so it can be located and fixed precisely."""


def build_auditor_llm():
    return ChatGroq(model="openai/gpt-oss-120b", temperature=0).with_structured_output(_GroundednessReviewSchema)


_auditor_llm = None


def compliance_auditor_node(state: SARWorkflowState) -> dict:
    """LangGraph node. Reads sar_draft + evidence_summary, writes state['compliance_check'] + trace."""
    global _auditor_llm
    if _auditor_llm is None:
        _auditor_llm = build_auditor_llm()

    start = time.perf_counter()
    draft = state["sar_draft"]
    evidence = state["evidence_summary"]
    regulatory = state["regulatory_findings"]

    completeness_issues = check_field_completeness(draft["structured_fields"])

    human_input = json.dumps(
        {
            "narrative": draft["narrative"],
            "consolidated_evidence": evidence["consolidated_evidence"],
            "regulatory_findings": regulatory,
        },
        indent=2,
        default=str,
    )

    messages = [
        ("system", GROUNDEDNESS_SYSTEM_PROMPT),
        ("human", f"Review this narrative against the evidence:\n{human_input}"),
    ]

    review: GroundednessReview = _auditor_llm.invoke( messages).model_dump()

    issues = list(completeness_issues)
    issues.extend(f"Ungrounded claim: {c}" for c in review["ungrounded_claims"])
    issues.extend(f"Missing narrative element ({e}): not addressed" for e in review["missing_narrative_elements"])

    result: ComplianceCheck = {
        "result": "pass" if not issues else "fail",
        "issues": issues,
    }
    duration_ms = (time.perf_counter() - start) * 1000

    trace = new_trace_entry(
        agent="Compliance Auditor",
        action=f"result={result['result']} ({len(issues)} issue(s))",
        input_summary=f"draft_version={draft['draft_version']}",
        output_summary="; ".join(issues) if issues else "No issues found.",
        metadata={
            "completeness_issue_count": len(completeness_issues),
            "groundedness_issue_count": len(review["ungrounded_claims"]),
        },
        duration_ms=duration_ms,
    )

    return {"compliance_check": result, "trace": [trace]}


# ---------------------------------------------------------------------------
# Quick manual test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    sample_draft = {
        "narrative": (
            "On the transaction date, Account ACC-A1 (USA) sent a wire transfer of "
            "$9,750 to Account ACC-B2 in the Cayman Islands. The amount is within "
            "$250 of the $10,000 currency transaction reporting threshold and "
            "deviates significantly (4.2 standard deviations) from the sender's "
            "typical transaction amount. The receiving account has only 3 prior "
            "transactions on record. Subject identity for both accounts is not "
            "established in available records. The activity is consistent with "
            "structuring to avoid reporting requirements, based on the proximity "
            "to the threshold and the cross-border movement of funds to an account "
            "with minimal prior history."
        ),
        "structured_fields": {
            "subject_name": "Unknown",
            "subject_address": "Unknown",
            "subject_tin_or_ssn": "Unknown",
            "subject_occupation_or_business_type": "Unknown",
            "subject_phone": "Unknown",
            "subject_relationship_to_institution": "Account holder",
            "activity_date_begin": "Unknown",
            "activity_date_end": "Unknown",
            "total_amount_involved": "$9,750",
            "suspicious_activity_type": "Structuring",
            "instrument_or_payment_mechanism": "Wire transfer",
            "institution_name": "Unknown",
            "institution_address": "Unknown",
            "institution_role": "Sender's bank",
            "preparer_name": "Unknown",
            "preparer_title": "Unknown",
            "preparer_phone": "Unknown",
            "date_prepared": "Unknown",
        },
        "draft_version": 1,
    }

    sample_evidence = {
        "consolidated_evidence": [
            "Transaction amount of $9,750 is within $250 of the $10,000 CTR reporting threshold.",
            "Transfer was cross-border, from a USA-based account to a Cayman Islands-based account.",
            "Transaction amount deviates 4.2 standard deviations from the sender's typical transaction amount.",
            "Receiving account has only 3 prior transactions on record.",
            "AML Auditor matched typology: structuring, with 0.82 confidence.",
        ],
        "evidence_sources": {},
    }

    test_state: SARWorkflowState = {
        "alert": {"transaction_id": "TXN-000123"},
        "investigator_brief": None,
        "aml_decision": None,
        "regulatory_findings": None,
        "evidence_summary": sample_evidence,
        "sar_draft": sample_draft,
        "compliance_check": None,
        "revision_count": 0,
        "trace": [],
        "status": "in_progress",
    }

    result = compliance_auditor_node(test_state)
    print(json.dumps(result, indent=2, default=str))