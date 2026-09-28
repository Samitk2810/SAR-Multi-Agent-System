"""
The full LangGraph StateGraph wiring together all nine pieces built so far.

    START
      |
   Investigator
      |
   AML Auditor ---- dismiss ----> END
      |
    escalate
      |
      +-----------------+
      |                 |
 Regulatory Analyst  Evidence Analyst      (parallel -- separate state keys,
      |                 |                   no merge conflict on join)
      +--------+--------+
               |
          SAR Drafter <-------------------+
               |                          |
        Compliance Auditor                |
          |         |                     |
        PASS       FAIL (capped) ---------+
          |          |
          |    revision_count >= MAX -> Max Revisions Exceeded -> END
          |
      Human Review (interrupt)
       /      |       \\
  approve  revise    reject
     |        |          |
    END   back to    END
  (filed) SAR Drafter  (dismissed, also capped)
         (same cap)

Run this file directly for an end-to-end demo, including the human-review
pause/resume.
"""

from __future__ import annotations

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from graph.sar_workflow_state import SARWorkflowState, new_trace_entry
from agents.investigator import investigator_node
from agents.aml_auditor import aml_auditor_node
from agents.regulatory_analyst_agent import regulatory_analyst_node
from agents.evidence_analyst import evidence_analyst_node
from agents.sar_drafter import sar_drafter_node
from agents.compliance_auditor import compliance_auditor_node
from agents.human_review import human_review_node


MAX_REVISIONS = 3


def mark_max_revisions_exceeded(state: SARWorkflowState) -> dict:
    """Terminal fallback when the FAIL <-> revision loop hits the cap without
    ever passing -- routes to a human queue for manual drafting rather than
    looping forever or silently giving up."""
    trace = new_trace_entry(
        agent="System",
        action="max revisions exceeded",
        input_summary=f"revision_count={state['revision_count']}",
        output_summary="Escalated for manual drafting -- automated pipeline could not produce a passing draft.",
    )
    return {"status": "max_revisions_exceeded", "trace": [trace]}


# ---------------------------------------------------------------------------
# Routing functions -- return node name(s) or END, read directly off state
# ---------------------------------------------------------------------------
def route_after_aml_auditor(state: SARWorkflowState):
    if state["status"] == "dismissed":
        return END
    # Returning a list fans out to both nodes in parallel -- they write to
    # separate state keys (regulatory_findings / evidence_summary), so there's
    # nothing to merge-conflict on when they both complete.
    return ["regulatory_analyst", "evidence_analyst"]


def route_after_compliance_auditor(state: SARWorkflowState):
    if state["compliance_check"]["result"] == "pass":
        return "human_review"
    if state["revision_count"] >= MAX_REVISIONS:
        return "max_revisions_exceeded"
    return "sar_drafter"  # same node, revision branch (see sar_drafter_agent.py)


def route_after_human_review(state: SARWorkflowState):
    if state["status"] == "in_progress":  # officer chose "revise"
        if state["revision_count"] >= MAX_REVISIONS:
            return "max_revisions_exceeded"
        return "sar_drafter"
    return END  # "filed" or "dismissed"


# ---------------------------------------------------------------------------
# Build the graph
# ---------------------------------------------------------------------------
def build_graph():
    builder = StateGraph(SARWorkflowState)

    builder.add_node("investigator", investigator_node)
    builder.add_node("aml_auditor", aml_auditor_node)
    builder.add_node("regulatory_analyst", regulatory_analyst_node)
    builder.add_node("evidence_analyst", evidence_analyst_node)
    builder.add_node("sar_drafter", sar_drafter_node)
    builder.add_node("compliance_auditor", compliance_auditor_node)
    builder.add_node("human_review", human_review_node)
    builder.add_node("max_revisions_exceeded", mark_max_revisions_exceeded)

    builder.add_edge(START, "investigator")
    builder.add_edge("investigator", "aml_auditor")

    builder.add_conditional_edges("aml_auditor", route_after_aml_auditor)

    # Fan-in: sar_drafter has two incoming edges and runs once both
    # predecessors finish -- no extra config needed, this is default
    # LangGraph superstep join behavior.
    builder.add_edge("regulatory_analyst", "sar_drafter")
    builder.add_edge("evidence_analyst", "sar_drafter")

    builder.add_edge("sar_drafter", "compliance_auditor")
    builder.add_conditional_edges("compliance_auditor", route_after_compliance_auditor)
    builder.add_conditional_edges("human_review", route_after_human_review)

    builder.add_edge("max_revisions_exceeded", END)

    # A checkpointer is REQUIRED -- human_review_node's interrupt() has
    # nothing to pause/resume across without one. MemorySaver is dev-only
    # (lost on process restart); swap for SqliteSaver/PostgresSaver for
    # anything a real compliance officer will review after a delay.
    checkpointer = MemorySaver()
    return builder.compile(checkpointer=checkpointer)


# ---------------------------------------------------------------------------
# End-to-end demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import json

    app = build_graph()

    # Optional: view the graph structure as Mermaid, e.g. to paste into
    # https://mermaid.live or render in a notebook.
    try:
        print(app.get_graph().draw_mermaid())
        print()
    except Exception:
        pass

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

    initial_state: SARWorkflowState = {
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

    config = {"configurable": {"thread_id": "TXN-000123"}}

    # Runs everything up through Compliance Auditor's PASS, then pauses at
    # human_review_node's interrupt().
    result = app.invoke(initial_state, config=config)

    if result.get("status") == "dismissed":
        print("Dismissed by AML Auditor -- no SAR drafted.")
        print(json.dumps(result["trace"], indent=2, default=str))
    elif "__interrupt__" in result:
        print("--- PAUSED FOR HUMAN REVIEW ---")
        payload = result["__interrupt__"][0].value
        print(f"Narrative:\n{payload['narrative']}\n")
        print(f"Compliance check: {payload['compliance_check']}\n")

        # Simulating a compliance officer's decision -- in a real deployment
        # this comes from whatever UI surfaces `payload` to a human, not a
        # hardcoded dict.
        final = app.invoke(Command(resume={"decision": "approve"}), config=config)
        print("--- RESUMED ---")
        print("Final status:", final["status"])
        print("\nFull trace:")
        for entry in final["trace"]:
            print(f"  [{entry['agent']}] {entry['action']}")