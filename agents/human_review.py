"""
Human Review node: the compliance-officer sign-off gate before a SAR is
actually filed. This sits AFTER Compliance Auditor's PASS, not inside it --
Compliance Auditor checks mechanical correctness (completeness, groundedness);
this step is a human judgment call on top of a draft that already passed those
checks. A filed SAR is a legally consequential document, so an automated PASS
should never be the last word.

Uses LangGraph's interrupt() -- this pauses graph execution and returns the
review_payload to whatever is driving the graph (a web UI, a CLI, a Slack bot).
Execution resumes only when that caller invokes the graph again with
Command(resume=<officer's decision>).

REQUIRES a checkpointer compiled into the graph (e.g. MemorySaver for local
dev, a persistent one like SqliteSaver/PostgresSaver for anything real) --
interrupt() has nothing to pause/resume across without one. See the __main__
block below for the minimal wiring.
"""

from __future__ import annotations
import time

from langgraph.types import interrupt

from graph.sar_workflow_state import SARWorkflowState, new_trace_entry


def human_review_node(state: SARWorkflowState) -> dict:
    """LangGraph node. Pauses for a compliance officer's decision, then routes
    based on it. Reads sar_draft/evidence_summary/regulatory_findings/aml_decision,
    writes trace + status (+ compliance_check/revision_count if sent back)."""
    draft = state["sar_draft"]
    evidence = state["evidence_summary"]
    regulatory = state["regulatory_findings"]
    aml = state.get("aml_decision")

    review_payload = {
        "transaction_id": state["alert"]["transaction_id"],
        "draft_version": draft["draft_version"],
        "narrative": draft["narrative"],
        "structured_fields": draft["structured_fields"],
        "matched_typology": aml["matched_typology"] if aml else None,
        "aml_confidence": aml["confidence"] if aml else None,
        "regulatory_findings": regulatory,
        "consolidated_evidence": evidence["consolidated_evidence"],
        "compliance_check": state["compliance_check"],
        "trace_so_far": state["trace"],  # full agent handoff history for the officer to review
        "instructions": (
            "Resume with one of:\n"
            "  {'decision': 'approve'}\n"
            "  {'decision': 'reject', 'reason': '...'}   -- dismiss, do not file\n"
            "  {'decision': 'revise', 'reason': '...'}   -- send back to SAR Drafter"
        ),
    }

    # Execution pauses here. Whatever the resuming caller passes into
    # Command(resume=...) becomes the return value of this call.
    human_response = interrupt(review_payload)

    start = time.perf_counter()
    decision = human_response.get("decision")
    reason = human_response.get("reason", "")

    if decision == "approve":
        status = "filed"
        action = "approved for filing"
    elif decision == "reject":
        status = "dismissed"
        action = f"rejected -- not filed. Reason: {reason}" if reason else "rejected -- not filed"
    elif decision == "revise":
        status = "in_progress"
        action = f"sent back for revision. Reason: {reason}" if reason else "sent back for revision"
    else:
        raise ValueError(
            f"Unrecognized human decision: {decision!r} -- expected 'approve', 'reject', or 'revise'"
        )

    trace = new_trace_entry(
        agent="Human Reviewer",
        action=action,
        input_summary=f"reviewing draft_version={draft['draft_version']}",
        output_summary=reason or "No additional comments.",
        duration_ms=(time.perf_counter() - start) * 1000,
    )

    update = {"trace": [trace], "status": status}

    if decision == "revise":
        # Feed the officer's reason into compliance_check so it routes back
        # through SAR Drafter's existing revision branch exactly like an
        # automated FAIL would -- no separate "human rejection" code path
        # needed in SAR Drafter.
        update["compliance_check"] = {
            "result": "fail",
            "issues": [f"Human reviewer requested revision: {reason}" if reason else "Human reviewer requested revision."],
        }
        update["revision_count"] = 1  # reducer -- counts toward the same cap as automated revisions

    return update


# ---------------------------------------------------------------------------
# Minimal wiring example -- shows the interrupt/resume mechanics in isolation,
# without the full graph. Run this file directly.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from langgraph.graph import StateGraph, START, END
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.types import Command

    # A trivial one-node graph just to demonstrate interrupt/resume -- your
    # real graph wires human_review_node in after Compliance Auditor's PASS.
    builder = StateGraph(SARWorkflowState)
    builder.add_node("human_review", human_review_node)
    builder.add_edge(START, "human_review")
    builder.add_edge("human_review", END)

    checkpointer = MemorySaver()
    graph = builder.compile(checkpointer=checkpointer)

    config = {"configurable": {"thread_id": "demo-thread-1"}}

    test_state: SARWorkflowState = {
        "alert": {"transaction_id": "TXN-000123"},
        "investigator_brief": None,
        "aml_decision": {
            "decision": "escalate", "matched_typology": "structuring",
            "confidence": 0.82, "reasoning": "N/A for this demo",
        },
        "regulatory_findings": {
            "threshold_triggered": True, "sanctions_hit": False, "pep_hit": False,
            "jurisdiction_risk": "medium", "notes": "N/A for this demo",
        },
        "evidence_summary": {
            "consolidated_evidence": ["Transaction amount of $9,750 is near the $10,000 CTR threshold."],
            "evidence_sources": {},
        },
        "sar_draft": {
            "narrative": "Sample narrative for demo purposes.",
            "structured_fields": {}, "draft_version": 1,
        },
        "compliance_check": {"result": "pass", "issues": []},
        "revision_count": 0,
        "trace": [],
        "status": "in_progress",
    }

    # First call runs until interrupt() pauses it, and surfaces review_payload.
    result = graph.invoke(test_state, config=config)
    print("--- PAUSED FOR HUMAN REVIEW ---")
    print(result["__interrupt__"])

    # Simulating a compliance officer approving it. In a real app this comes
    # from your UI, not a hardcoded dict.
    final = graph.invoke(Command(resume={"decision": "approve"}), config=config)
    print("\n--- RESUMED ---")
    print("status:", final["status"])
    print("trace:", final["trace"])