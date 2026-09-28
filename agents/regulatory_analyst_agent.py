"""
Regulatory Analyst agent: parallel branch A (runs alongside Evidence Analyst
after AML Auditor escalates).

Two tools, both used for the parts an LLM should NOT be guessing at:

1. OpenSanctions /match  -- real sanctions/PEP entity screening.
   Docs: https://www.opensanctions.org/docs/api/quickstart/
   Needs OPENSANCTIONS_API_KEY (free non-commercial key: opensanctions.org/account).
   NOTE: this endpoint screens by NAME. Your current TransactionAlert schema only
   has account IDs and bank *locations*, not counterparty names, so this function
   is wired and tested but will no-op until sender_name/receiver_name are available
   somewhere in your pipeline (e.g. added to the alert, or looked up separately).

2. Tavily Search -- live jurisdiction risk context (current FATF grey/black list
   status, recent advisories), replacing a hardcoded, staleness-prone country list.
   Needs TAVILY_API_KEY (free tier: tavily.com).

Compliance-critical booleans (sanctions_hit, pep_hit, threshold_triggered) are still
derived from tool output / plain code, never from LLM judgment. The LLM's only job
is writing the `notes` narrative from what the tools + upstream context established.
"""

from __future__ import annotations
import os
import time
import json
import requests
from dotenv import load_dotenv
load_dotenv()

from langchain_openrouter import ChatOpenRouter

from graph.sar_workflow_state import (
    SARWorkflowState,
    TransactionAlert,
    RegulatoryFindings,
    new_trace_entry,
)


TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")
OPENSANCTIONS_API_KEY = os.environ.get("OPENSANCTIONS_API_KEY")

# Static fallback only -- used if Tavily is unavailable/rate-limited, or as a
# floor under the live search result. Keep it small; the live search is the
# real source of truth now.
HIGH_RISK_JURISDICTIONS_FALLBACK = {
    "Iran", "North Korea", "Myanmar", "Syria", "Cuba",
    "Cayman Islands", "Panama", "Cambodia",
}

RISK_KEYWORDS = (
    "grey list", "greylist", "gray list", "black list", "blacklist",
    "high-risk jurisdiction", "high risk jurisdiction", "increased monitoring",
    "call for action",
)


# ---------------------------------------------------------------------------
# Tool 1: Tavily -- live jurisdiction risk search
# ---------------------------------------------------------------------------
def tavily_jurisdiction_search(location: str, max_results: int = 3) -> list[dict]:
    """Search for current AML/FATF risk status of a jurisdiction. Returns
    [{"title", "url", "content"}, ...]. Returns [] on missing key or error --
    callers should fall back to the static list, not crash the pipeline."""
    if not TAVILY_API_KEY:
        return []
    try:
        resp = requests.post(
            "https://api.tavily.com/search",
            json={
                "api_key": TAVILY_API_KEY,
                "query": f"{location} FATF grey list high risk jurisdiction AML 2026",
                "search_depth": "basic",
                "max_results": max_results,
            },
            timeout=15,
        )
        resp.raise_for_status()
        results = resp.json().get("results", [])
        return [{"title": r.get("title", ""), "url": r.get("url", ""), "content": r.get("content", "")} for r in results]
    except requests.RequestException as e:
        print(f"[tavily_jurisdiction_search] request failed: {e}")
        return []


def _live_risk_flag(location: str) -> tuple[bool, list[dict]]:
    """True if live search results suggest this location is currently
    grey/black-listed or under increased monitoring."""
    results = tavily_jurisdiction_search(location)
    combined_text = " ".join(r["content"].lower() for r in results)
    flagged = any(kw in combined_text for kw in RISK_KEYWORDS)
    return flagged, results


# ---------------------------------------------------------------------------
# Tool 2: OpenSanctions -- real sanctions/PEP entity screening
# ---------------------------------------------------------------------------
def opensanctions_screen(name: str, schema: str = "LegalEntity") -> dict:
    """Screen a name against OpenSanctions (sanctions + PEP data combined).
    `schema="LegalEntity"` matches broadly across people and organizations
    since it's the common base type in OpenSanctions' entity model.

    Returns {"query": name, "hit": bool, "topics": [...], "top_match": {...}|None}.
    Returns hit=False (not a silent pass -- an explicit "screened, no match")
    on missing key or request failure, and logs the failure -- a screening
    call that silently fails should never be indistinguishable from a clean
    result in a compliance system.
    """
    if not name:
        return {"query": None, "hit": False, "topics": [], "top_match": None, "skipped_reason": "no name provided"}

    if not OPENSANCTIONS_API_KEY:
        return {"query": name, "hit": False, "topics": [], "top_match": None, "skipped_reason": "no API key configured"}

    try:
        session = requests.Session()
        session.headers["Authorization"] = f"ApiKey {OPENSANCTIONS_API_KEY}"
        resp = session.post(
            "https://api.opensanctions.org/match/default",
            json={"queries": {"q": {"schema": schema, "properties": {"name": [name]}}}},
            timeout=15,
        )
        resp.raise_for_status()
        results = resp.json()["responses"]["q"]["results"]
    except (requests.RequestException, KeyError) as e:
        print(f"[opensanctions_screen] request failed for '{name}': {e}")
        return {"query": name, "hit": False, "topics": [], "top_match": None, "skipped_reason": f"request error: {e}"}

    if not results:
        return {"query": name, "hit": False, "topics": [], "top_match": None}

    top = results[0]
    topics = top.get("properties", {}).get("topics", [])
    hit = top.get("match", False) or "sanction" in topics or "role.pep" in topics

    return {
        "query": name,
        "hit": hit,
        "topics": topics,
        "top_match": {"caption": top.get("caption"), "score": top.get("score"), "datasets": top.get("datasets", [])},
    }


# ---------------------------------------------------------------------------
# Deterministic assembly -- combines tool output, no LLM involved
# ---------------------------------------------------------------------------
def check_sanctions_and_jurisdiction(
    alert: TransactionAlert,
    sender_name: str | None = None,
    receiver_name: str | None = None,
) -> dict:
    sender_loc = alert["sender_bank_location"]
    receiver_loc = alert["receiver_bank_location"]

    sender_screen = opensanctions_screen(sender_name) if sender_name else {"hit": False, "skipped_reason": "no sender name in alert"}
    receiver_screen = opensanctions_screen(receiver_name) if receiver_name else {"hit": False, "skipped_reason": "no receiver name in alert"}
    sanctions_hit = bool(sender_screen.get("hit") or receiver_screen.get("hit"))
    pep_hit = "role.pep" in sender_screen.get("topics", []) or "role.pep" in receiver_screen.get("topics", [])

    sender_live_flag, sender_search = _live_risk_flag(sender_loc)
    receiver_live_flag, receiver_search = _live_risk_flag(receiver_loc)

    static_hits = {sender_loc, receiver_loc} & HIGH_RISK_JURISDICTIONS_FALLBACK
    live_hits = {loc for loc, flag in [(sender_loc, sender_live_flag), (receiver_loc, receiver_live_flag)] if flag}
    risky_locations = static_hits | live_hits

    if sanctions_hit or len(risky_locations) >= 2:
        jurisdiction_risk = "high"
    elif risky_locations:
        jurisdiction_risk = "medium"
    else:
        jurisdiction_risk = "low"

    return {
        "threshold_triggered": alert["near_reporting_threshold"] or alert["amount"] >= 10000,
        "sanctions_hit": sanctions_hit,
        "pep_hit": pep_hit,
        "jurisdiction_risk": jurisdiction_risk,
        "risky_locations": sorted(risky_locations),
        "sender_screen": sender_screen,
        "receiver_screen": receiver_screen,
        "jurisdiction_search_snippets": (sender_search + receiver_search)[:3],
    }


SYSTEM_PROMPT = """You are a Regulatory Analyst. You are given deterministic \
compliance screening results (OpenSanctions entity screening + live jurisdiction \
risk search -- already computed, do not second-guess or contradict them) plus the \
case context so far.

Write a concise, factual `notes` field explaining the regulatory picture: what \
reporting obligations are implicated, why the jurisdiction risk level was assigned \
(cite the search snippets if they informed it), and any regulatory context relevant \
to a compliance officer reviewing this case. Do not restate the raw booleans -- \
explain what they mean for this case. Do not invent sanctions or PEP hits beyond \
what was already determined, and do not treat a skipped screen (no name available) \
as a clean result -- note it as a gap if relevant."""


def build_regulatory_llm():
    return  ChatOpenRouter(
    model="liquid/lfm-2.5-2.6b:free",
    temperature=0,
)


_regulatory_llm = None


def regulatory_analyst_node(state: SARWorkflowState) -> dict:
    """LangGraph node. Reads state['alert'] + upstream context, writes state['regulatory_findings'] + trace."""
    global _regulatory_llm
    if _regulatory_llm is None:
        _regulatory_llm = build_regulatory_llm()

    start = time.perf_counter()
    alert = state["alert"]

    # sender_name/receiver_name aren't in the current TransactionAlert schema --
    # pulled via .get() so this works today (no-op) and picks up real values
    # automatically once your pipeline provides them.
    deterministic = check_sanctions_and_jurisdiction(
        alert,
        sender_name=alert.get("sender_name"),
        receiver_name=alert.get("receiver_name"),
    )

    human_input = json.dumps(
        {
            "deterministic_findings": deterministic,
            "aml_decision": state.get("aml_decision"),
        },
        indent=2,
        default=str,
    )

    messages = [
        ("system", SYSTEM_PROMPT),
        ("human", f"Case context:\n{human_input}"),
    ]

    notes = _regulatory_llm.invoke(messages).content
    duration_ms = (time.perf_counter() - start) * 1000

    findings: RegulatoryFindings = {
        "threshold_triggered": deterministic["threshold_triggered"],
        "sanctions_hit": deterministic["sanctions_hit"],
        "pep_hit": deterministic["pep_hit"],
        "jurisdiction_risk": deterministic["jurisdiction_risk"],
        "notes": notes,
    }

    trace = new_trace_entry(
        agent="Regulatory Analyst",
        action=f"jurisdiction_risk={findings['jurisdiction_risk']}, sanctions_hit={findings['sanctions_hit']}",
        input_summary=f"sender={alert['sender_bank_location']}, receiver={alert['receiver_bank_location']}",
        output_summary=notes,
        metadata={
            "risky_locations": deterministic["risky_locations"],
            "sender_screen_skipped": deterministic["sender_screen"].get("skipped_reason"),
            "receiver_screen_skipped": deterministic["receiver_screen"].get("skipped_reason"),
        },
        duration_ms=duration_ms,
    )

    return {"regulatory_findings": findings, "trace": [trace]}


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
        # Not part of the formal schema yet -- shown here to demonstrate
        # opensanctions_screen actually firing when a name IS available.
        "receiver_name": "Viktor Bout",
    }

    test_state: SARWorkflowState = {
        "alert": sample_alert,
        "investigator_brief": None,
        "aml_decision": {
            "decision": "escalate",
            "matched_typology": "structuring",
            "confidence": 0.82,
            "reasoning": "Amount just under reporting threshold combined with cross-border transfer to high-risk jurisdiction.",
        },
        "regulatory_findings": None,
        "evidence_summary": None,
        "sar_draft": None,
        "compliance_check": None,
        "revision_count": 0,
        "trace": [],
        "status": "in_progress",
    }

    result = regulatory_analyst_node(test_state)
    print(json.dumps(result, indent=2, default=str))
