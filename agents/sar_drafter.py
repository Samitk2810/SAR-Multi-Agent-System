"""
SAR Drafter agent: drafts the actual FinCEN SAR narrative + structured fields,
using ONLY regulatory_findings + evidence_summary as input -- not the raw alert
or investigator_brief directly. This is deliberate: it forces every fact in the
draft to trace back through the Evidence Analyst's already-sourced claims, which
is what the Compliance Auditor's groundedness check verifies later.

Two tools:

1. get_fincen_sar_template() -- static, no API. Returns the real FinCEN SAR
   structure (Parts I-V) so structured_fields uses FinCEN's actual section names
   instead of the model inventing its own field names. Grounded in FinCEN's SAR
   Electronic Filing Instructions and the FFIEC BSA/AML Exam Manual.
   Includes the "Unknown" convention FinCEN actually uses: critical fields that
   can't be filled get an explicit unknown flag rather than "N/A" or blank.

2. tavily_fincen_advisory_search(typology) -- live search for current FinCEN
   advisories and SAR narrative key terms tied to the matched typology. FinCEN
   periodically issues advisories with specific key terms it wants included in
   narratives (e.g. FIN-2023-xxx), and guidance changes over time (there was a
   real shift in FinCEN's narrative-quality expectations as recently as October
   2025) -- static hardcoded terms would go stale, so this is looked up live.
"""

from __future__ import annotations
import os
import time
import json
import requests

from pydantic import BaseModel, create_model
from langchain_openrouter import ChatOpenRouter

from graph.sar_workflow_state import (
    SARWorkflowState,
    SARDraft,
    new_trace_entry,
)


def _build_sar_draft_schema():
    """Dynamically builds a Pydantic model with one REQUIRED string field per
    FinCEN template key (Parts I-IV), sourced from get_fincen_sar_template()
    so this can't drift out of sync with Compliance Auditor's completeness
    check. This replaces a generic `structured_fields: dict[str, str]` field,
    which gave the model no structural signal about which keys to produce --
    some models (seen with Llama 3.3 via Groq) simply return an empty dict
    for that shape despite prose instructions. Making each field required in
    the schema itself means the API-level constrained decoding enforces
    their presence, not just a prompt asking nicely."""
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
    fields = {key: (str, ...) for key in required_keys}
    fields["narrative"] = (str, ...)
    return create_model("_SARDraftSchema", **fields), required_keys





TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")


# ---------------------------------------------------------------------------
# Tool 1: FinCEN SAR structure reference (static, no API needed)
# ---------------------------------------------------------------------------
def get_fincen_sar_template() -> dict:
    """Real FinCEN SAR structure. Source: FinCEN SAR Electronic Filing
    Instructions / FFIEC BSA-AML Examination Manual. Use these exact section
    keys when populating SARDraft.structured_fields, and use "Unknown" as the
    value for critical fields you don't have data for -- FinCEN's actual form
    has a dedicated "Unknown" checkbox for exactly this, rather than blank or
    "N/A"."""
    return {
        "part_i_subject_information": [
            "subject_name", "subject_address", "subject_tin_or_ssn",
            "subject_occupation_or_business_type", "subject_phone",
            "subject_relationship_to_institution",
        ],
        "part_ii_suspicious_activity_information": [
            "activity_date_begin", "activity_date_end", "total_amount_involved",
            "suspicious_activity_type", "instrument_or_payment_mechanism",
        ],
        "part_iii_financial_institution_information": [
            "institution_name", "institution_address", "institution_role",
        ],
        "part_iv_filing_institution_contact_information": [
            "preparer_name", "preparer_title", "preparer_phone", "date_prepared",
        ],
        "part_v_narrative_requirements": [
            "who", "what", "when", "where", "why", "how",
        ],
        "note": (
            "Critical fields with no available data should be set to the literal "
            "string 'Unknown' -- do not leave blank or write 'N/A'/'XX'."
        ),
    }

_SARDraftSchema, _FINCEN_FIELD_KEYS = _build_sar_draft_schema()

# ---------------------------------------------------------------------------
# Tool 2: Tavily -- live FinCEN advisory / key-term search
# ---------------------------------------------------------------------------
def tavily_fincen_advisory_search(typology: str, max_results: int = 3) -> list[dict]:
    """Search for current FinCEN advisories and SAR narrative key terms tied to
    a laundering typology. Returns [{"title","url","content"}, ...], or [] if
    unavailable -- caller should draft without advisory context in that case,
    not fail the whole node."""
    if not TAVILY_API_KEY or not typology:
        return []
    try:
        resp = requests.post(
            "https://api.tavily.com/search",
            json={
                "api_key": TAVILY_API_KEY,
                "query": f"FinCEN advisory SAR narrative key terms {typology}",
                "search_depth": "basic",
                "max_results": max_results,
            },
            timeout=15,
        )
        resp.raise_for_status()
        results = resp.json().get("results", [])
        return [{"title": r.get("title", ""), "url": r.get("url", ""), "content": r.get("content", "")} for r in results]
    except requests.RequestException as e:
        print(f"[tavily_fincen_advisory_search] request failed: {e}")
        return []


INITIAL_SYSTEM_PROMPT = """You are drafting a FinCEN Suspicious Activity Report (SAR).

You are given:
1. A vetted evidence package (consolidated_evidence + evidence_sources) -- this is
   your ONLY source of factual claims. Do not introduce any fact, number, date, or
   detail that isn't traceable to one of these items.
2. Regulatory findings (sanctions/PEP/jurisdiction/threshold status).
3. The FinCEN SAR template -- use these exact section keys in structured_fields.
4. Optionally, current FinCEN advisory context relevant to the matched typology --
   use this only to inform terminology/framing, never as a source of case facts.

Requirements:
- The narrative (Part V) MUST address all six elements FinCEN requires: who, what,
  when, where, why, and how. If evidence is missing for one of these, say so
  explicitly rather than inventing detail to fill the gap (e.g. "Subject identity
  is not established in available records").
- Write in plain, factual, third-person language. No speculation, no hedging like
  "may have" beyond what the evidence itself supports, no dramatic language.
- Populate structured_fields using the template's exact keys. Use "Unknown" for
  critical fields (subject_name, subject_tin_or_ssn, etc.) when the evidence
  package doesn't contain that information -- this is standard FinCEN practice,
  not a flaw in the draft.
- Every claim in the narrative must map to something in consolidated_evidence or
  regulatory_findings. This will be checked mechanically by the Compliance Auditor next.
- Do NOT add recommendations, directives, or action items (e.g. "the institution
  should conduct enhanced due diligence") unless evidence_summary or
  regulatory_findings explicitly states that action was taken or required. A SAR
  narrative documents what happened and why it's being reported -- it does not
  issue instructions."""


REVISION_SYSTEM_PROMPT = """You are revising a FinCEN SAR draft that FAILED
compliance review. You are given the previous draft, the specific issues the
Compliance Auditor raised, and the same vetted evidence package used originally.

Rules:
- Fix ONLY what the issues describe. Do not rewrite sections that weren't
  flagged, and do not introduce any new fact, number, or detail beyond what's
  already in consolidated_evidence -- that would just create new groundedness
  failures on the next review pass.
- For an "Ungrounded claim" issue: remove or rephrase the flagged sentence so
  it only asserts what evidence actually supports, or delete it entirely if no
  version of it is supportable.
- For a "Missing narrative element" issue: add a sentence addressing that
  element using only evidence-backed facts, or explicitly state it's
  unknown/unavailable if that's genuinely what the evidence shows.
- For a "Missing required field" / "empty field" issue: fill the exact
  structured_fields key named in the issue -- use "Unknown" if the evidence
  package has no relevant data for it.
- Keep everything else from the previous draft unchanged."""


def build_drafter_llm():
    return ChatOpenRouter(model="liquid/lfm-2.5-2.6b:free", temperature=0).with_structured_output(_SARDraftSchema)


_drafter_llm = None


def sar_drafter_node(state: SARWorkflowState) -> dict:
    """LangGraph node -- handles BOTH the initial draft and every revision pass.

    Which mode runs is decided by state, not by which node called it: if
    compliance_check is present and its result is "fail", this is a revision
    (fix the named issues, evidence stays the source of truth). Otherwise it's
    a fresh draft from evidence_summary + regulatory_findings.

    This is also why the graph's FAIL edge can point straight back to this same
    node instead of a separate Revision node -- the state already carries
    everything needed to tell the two modes apart.
    """
    global _drafter_llm
    if _drafter_llm is None:
        _drafter_llm = build_drafter_llm()

    start = time.perf_counter()
    evidence = state["evidence_summary"]
    regulatory = state["regulatory_findings"]
    prior_check = state.get("compliance_check")
    is_revision = bool(prior_check and prior_check["result"] == "fail")
    template = get_fincen_sar_template()

    if is_revision:
        prior_draft = state["sar_draft"]
        human_payload = {
            "evidence_summary": evidence,
            "regulatory_findings": regulatory,
            "fincen_sar_template": template,
            "previous_draft": prior_draft,
            "compliance_issues_to_fix": prior_check["issues"],
        }
        system_prompt = REVISION_SYSTEM_PROMPT
        action_verb = "Revise"
    else:
        # matched_typology is only used to focus the advisory search -- it is
        # NOT injected into the draft as a fact beyond what evidence_summary
        # already established (Evidence Analyst already captured it as a
        # sourced claim). Skipped entirely on revisions -- framing context
        # doesn't need re-fetching just to fix a flagged sentence.
        typology = (state.get("aml_decision") or {}).get("matched_typology")
        advisory_results = tavily_fincen_advisory_search(typology) if typology else []
        human_payload = {
            "evidence_summary": evidence,
            "regulatory_findings": regulatory,
            "fincen_sar_template": template,
            "relevant_fincen_advisory_context": advisory_results,
        }
        system_prompt = INITIAL_SYSTEM_PROMPT
        action_verb = "Draft"

    messages = [
        ("system", system_prompt),
        ("human", f"{action_verb} the SAR from this case record:\n{json.dumps(human_payload, indent=2, default=str)}"),
    ]

    raw = _drafter_llm.invoke( messages).model_dump()
    narrative = raw.pop("narrative")
    draft: SARDraft = {
        "narrative": narrative,
        "structured_fields": raw,  # every remaining key is one of _FINCEN_FIELD_KEYS
        "draft_version": (prior_check and state["sar_draft"]["draft_version"] + 1) or 1,
    }
    duration_ms = (time.perf_counter() - start) * 1000

    trace = new_trace_entry(
        agent="SAR Drafter",
        action=f"drafted v{draft['draft_version']}"
        + (f" (revision, fixed {len(prior_check['issues'])} issue(s))" if is_revision else " (initial)"),
        input_summary=(
            f"revising after {len(prior_check['issues'])} compliance issue(s)"
            if is_revision
            else f"{len(evidence['consolidated_evidence'])} evidence items, jurisdiction_risk={regulatory['jurisdiction_risk']}"
        ),
        output_summary=draft["narrative"][:300] + ("..." if len(draft["narrative"]) > 300 else ""),
        duration_ms=duration_ms,
    )

    update = {"sar_draft": draft, "trace": [trace]}
    if is_revision:
        update["revision_count"] = 1  # reducer (operator.add) sums this across passes
    return update


# ---------------------------------------------------------------------------
# Quick manual test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    sample_evidence = {
        "consolidated_evidence": [
            "Transaction amount of $9,750 is within $250 of the $10,000 CTR reporting threshold.",
            "Transfer was cross-border, from a USA-based account to a Cayman Islands-based account.",
            "Transaction amount deviates 4.2 standard deviations from the sender's typical transaction amount.",
            "Receiving account has only 3 prior transactions on record.",
            "AML Auditor matched typology: structuring, with 0.82 confidence.",
        ],
        "evidence_sources": {
            "Transaction amount of $9,750 is within $250 of the $10,000 CTR reporting threshold.": "alert.amount, alert.near_reporting_threshold",
            "Transfer was cross-border, from a USA-based account to a Cayman Islands-based account.": "alert.sender_bank_location, alert.receiver_bank_location",
            "Transaction amount deviates 4.2 standard deviations from the sender's typical transaction amount.": "alert.amount_dev_from_sender_avg",
            "Receiving account has only 3 prior transactions on record.": "alert.receiver_txn_count",
            "AML Auditor matched typology: structuring, with 0.82 confidence.": "AML Auditor: matched_typology, confidence",
        },
    }

    sample_regulatory = {
        "threshold_triggered": True,
        "sanctions_hit": False,
        "pep_hit": False,
        "jurisdiction_risk": "medium",
        "notes": "Cayman Islands carries elevated jurisdiction risk per current FATF monitoring status. No sanctions or PEP hits identified; counterparty names were not available for screening.",
    }

    test_state: SARWorkflowState = {
        "alert": {"transaction_id": "TXN-000123"},  # not read by this node beyond trace context
        "investigator_brief": None,
        "aml_decision": {
            "decision": "escalate",
            "matched_typology": "structuring",
            "confidence": 0.82,
            "reasoning": "N/A for this test",
        },
        "regulatory_findings": sample_regulatory,
        "evidence_summary": sample_evidence,
        "sar_draft": None,
        "compliance_check": None,
        "revision_count": 0,
        "trace": [],
        "status": "in_progress",
    }

    result = sar_drafter_node(test_state)
    print(json.dumps(result, indent=2, default=str))