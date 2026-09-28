"""
Entry point: loads every XGBoost-flagged transaction and runs them through the
SAR workflow graph CONCURRENTLY, up to MAX_CONCURRENT_CASES at a time.

Each alert is an independent graph run (its own thread_id in the checkpointer),
not to be confused with the internal Regulatory/Evidence parallel fan-out that
happens *inside* a single alert's run -- this is a separate, outer layer of
concurrency across different alerts.

Any case that reaches Human Review will pause there (interrupt()) and show up
in the batch results as "awaiting_human_review" rather than blocking the rest
of the batch -- resuming those is a separate step (e.g. a review UI/CLI that
calls Command(resume=...) against that case's thread_id).
"""

import asyncio
from pathlib import Path

import pandas as pd

from graph.sar_workflow_state import SARWorkflowState, TransactionAlert
from graph.sar_graph import build_graph


BASE_DIR = Path(__file__).resolve().parent
SCREENING_PATH = BASE_DIR / "data" / "screening_results.csv"

# Gemini / Tavily / OpenSanctions free tiers all have per-minute rate limits --
# bound how many cases run through the graph at once rather than firing every
# alert simultaneously. Tune based on whatever your tightest API limit is.
MAX_CONCURRENT_CASES = 3


def _row_to_alert(row: pd.Series) -> TransactionAlert:
    return {
        "transaction_id": str(row.name),
        "sender_account": str(row["Sender_account"]),
        "receiver_account": str(row["Receiver_account"]),
        "amount": float(row["Amount"]),
        "payment_type": str(row["Payment_type"]),
        "payment_currency": str(row["Payment_currency"]),
        "received_currency": str(row["Received_currency"]),
        "sender_bank_location": str(row["Sender_bank_location"]),
        "receiver_bank_location": str(row["Receiver_bank_location"]),
        "transaction_date": str(row["Date"]),
        "transaction_time": str(row["Time"]),

        # XGBoost output
        "xgb_score": float(row["risk_probability"]),
        "xgb_top_features": [],

        # Engineered features
        "sender_txn_count": int(row["Sender_txn_count"]),
        "receiver_txn_count": int(row["Receiver_txn_count"]),
        "amount_dev_from_sender_avg": float(row["Amount_dev_from_sender_avg"]),
        "is_cross_border": bool(row["Is_cross_border"]),
        "is_round_amount": bool(row["Is_round_amount"]),
        "near_reporting_threshold": bool(row["Near_reporting_threshold"]),
    }


def load_suspicious_alerts() -> list[TransactionAlert]:
    df = pd.read_csv(SCREENING_PATH)

    # Keep only transactions flagged by XGBoost
    suspicious = df[df["sar_worthy"] == 1]

    if suspicious.empty:
        raise ValueError("No suspicious transactions found.")

    return [_row_to_alert(row) for _, row in suspicious.iterrows()]


def create_initial_state(alert: TransactionAlert) -> SARWorkflowState:
    return {
        "alert": alert,
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


async def process_alert(app, alert: TransactionAlert, semaphore: asyncio.Semaphore) -> dict:
    """Runs one alert through the graph up to its first stopping point
    (dismissed by AML Auditor, paused at Human Review, or an error). Never
    raises -- one bad case shouldn't take down the rest of the batch."""
    async with semaphore:
        state = create_initial_state(alert)
        # thread_id must be unique per case -- it's what lets a paused
        # Human Review interrupt be resumed against the right case later.
        config = {"configurable": {"thread_id": alert["transaction_id"]}}

        try:
            result = await app.ainvoke(state, config=config)
        except Exception as e:
            return {
                "transaction_id": alert["transaction_id"],
                "outcome": "error",
                "detail": str(e),
            }

        if "__interrupt__" in result:
            return {
                "transaction_id": alert["transaction_id"],
                "outcome": "awaiting_human_review",
                "thread_id": alert["transaction_id"],
                "draft_version": result["sar_draft"]["draft_version"],
            }

        return {
            "transaction_id": alert["transaction_id"],
            "outcome": result["status"],  # "dismissed", "max_revisions_exceeded", etc.
            "trace_steps": len(result["trace"]),
            "final_compliance_issues": (
                result["compliance_check"]["issues"] if result.get("compliance_check") else None
            ),
            "trace": result["trace"],
        }


async def process_all(alerts: list[TransactionAlert]) -> list[dict]:
    app = build_graph()
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_CASES)
    tasks = [process_alert(app, alert, semaphore) for alert in alerts]
    return await asyncio.gather(*tasks)


def main():
    alerts = load_suspicious_alerts()
    print(
        f"Loaded {len(alerts)} suspicious transaction(s). "
        f"Processing up to {MAX_CONCURRENT_CASES} concurrently...\n"
    )

    results = asyncio.run(process_all(alerts))

    print("--- Batch results ---")
    for r in results:
        print({k: v for k, v in r.items() if k != "trace"})

    unresolved = [r for r in results if r["outcome"] == "max_revisions_exceeded"]
    if unresolved:
        print("\n--- Compliance Auditor history for unresolved case(s) ---")
        for r in unresolved:
            print(f"\ntransaction_id={r['transaction_id']}:")
            for entry in r["trace"]:
                if entry["agent"] in ("SAR Drafter", "Compliance Auditor"):
                    print(f"  [{entry['agent']}] {entry['action']}")
                    print(f"    -> {entry['output_summary']}")

    awaiting = [r for r in results if r["outcome"] == "awaiting_human_review"]
    errored = [r for r in results if r["outcome"] == "error"]

    if awaiting:
        print(
            f"\n{len(awaiting)} case(s) awaiting human review. "
            f"Resume each using its thread_id (== transaction_id) with "
            f"Command(resume={{'decision': ...}})."
        )
    if errored:
        print(f"\n{len(errored)} case(s) errored -- see 'detail' above.")


if __name__ == "__main__":
    main()