"""FastAPI backend for uploaded AML screening and the existing SAR graph.

Screening uploads, screening results, and LangGraph MemorySaver state are
process-local. Restarting this API loses them, and paused graph threads cannot
be recovered by another process. A persistent store/checkpointer belongs in a
separate production change; this module intentionally keeps the existing
MemorySaver architecture unchanged.
"""

from __future__ import annotations

import asyncio
import io
import pickle
import uuid
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from langgraph.types import Command
from pydantic import BaseModel, Field, field_validator

from graph.sar_graph import build_graph
from main import create_initial_state


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "ml" / "models" / "xgb_sar_model.pkl"
SCREENING_THRESHOLD = 0.87
REQUIRED_COLUMNS = {
    "Time",
    "Date",
    "Sender_account",
    "Receiver_account",
    "Amount",
    "Payment_currency",
    "Received_currency",
    "Sender_bank_location",
    "Receiver_bank_location",
    "Payment_type",
}
MAX_CASE_CONCURRENCY = 3

app = FastAPI(title="AML SAR Compliance API", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

with MODEL_PATH.open("rb") as model_file:
    _model_package = pickle.load(model_file)
_model = _model_package["model"]
_model_feature_columns = _model_package["feature_columns"]
_model_categorical_columns = _model_package["categorical_columns"]
_category_mappings = _model_package["category_mappings"]

# These are deliberately in-memory to preserve the existing project design.
_screenings: dict[str, dict[str, Any]] = {}
_cases: dict[str, dict[str, Any]] = {}
_graph = build_graph()
_screen_lock = asyncio.Lock()
_case_run_lock = asyncio.Lock()


class ReasonRequest(BaseModel):
    reason: str = Field(..., min_length=1)

    @field_validator("reason")
    @classmethod
    def reason_must_contain_text(cls, reason: str) -> str:
        if not reason.strip():
            raise ValueError("reason must not be empty")
        return reason.strip()


class RunCasesRequest(BaseModel):
    screening_id: str = Field(..., min_length=1)
    transaction_ids: list[str] = Field(..., min_length=1)

    @field_validator("transaction_ids")
    @classmethod
    def transaction_ids_must_be_unique(cls, ids: list[str]) -> list[str]:
        cleaned = [str(transaction_id).strip() for transaction_id in ids]
        if not all(cleaned):
            raise ValueError("transaction_ids must not contain empty values")
        if len(cleaned) != len(set(cleaned)):
            raise ValueError("transaction_ids must be unique")
        return cleaned


def _json_value(value: Any) -> Any:
    if pd.isna(value) if not isinstance(value, (dict, list, tuple)) else False:
        return None
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat()
    if hasattr(value, "item"):
        return value.item()
    return value


def _config(transaction_id: str) -> dict[str, dict[str, str]]:
    return {"configurable": {"thread_id": transaction_id}}


def _has_interrupt(snapshot: Any) -> bool:
    tasks = getattr(snapshot, "tasks", ()) or ()
    if any(getattr(task, "interrupts", ()) for task in tasks):
        return True
    values = getattr(snapshot, "values", {}) or {}
    return "__interrupt__" in values


def _state_for(transaction_id: str) -> dict[str, Any] | None:
    try:
        snapshot = _graph.get_state(_config(transaction_id))
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Unable to read workflow state.") from exc
    values = getattr(snapshot, "values", {}) or {}
    if not values:
        return None
    state = dict(values)
    state["transaction_id"] = transaction_id
    state["status"] = "awaiting_human_review" if _has_interrupt(snapshot) else state.get("status", "unknown")
    return state


def _case_metadata(transaction_id: str) -> dict[str, Any]:
    return _cases.get(transaction_id, {})


def _require_case(transaction_id: str) -> dict[str, Any]:
    if transaction_id not in _cases:
        raise HTTPException(status_code=404, detail="Case not found in this API process.")
    state = _state_for(transaction_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Case state is not available.")
    return state


def _status_summary(transaction_id: str, state: dict[str, Any]) -> dict[str, Any]:
    draft = state.get("sar_draft") or {}
    metadata = _case_metadata(transaction_id)
    return {
        "transaction_id": transaction_id,
        "status": state.get("status", "unknown"),
        "draft_version": draft.get("draft_version"),
        "screening_id": metadata.get("screening_id"),
    }


def _case_response(state: dict[str, Any]) -> dict[str, Any]:
    alert = state.get("alert") or {}
    investigator = state.get("investigator_brief") or {}
    aml = state.get("aml_decision") or {}
    regulatory = state.get("regulatory_findings") or {}
    evidence = state.get("evidence_summary") or {}
    draft = state.get("sar_draft") or {}
    return {
        "transaction_id": state["transaction_id"],
        "transaction_date": alert.get("transaction_date"),
        "transaction_time": alert.get("transaction_time"),
        "sender_account": alert.get("sender_account"),
        "receiver_account": alert.get("receiver_account"),
        "amount": alert.get("amount"),
        "payment_type": alert.get("payment_type"),
        "payment_currency": alert.get("payment_currency"),
        "received_currency": alert.get("received_currency"),
        "sender_bank_location": alert.get("sender_bank_location"),
        "receiver_bank_location": alert.get("receiver_bank_location"),
        "xgb_score": alert.get("xgb_score"),
        "investigator": investigator,
        "anomaly_summary": investigator.get("anomaly_summary"),
        "flagged_signals": investigator.get("flagged_signals", []),
        "initial_risk_level": investigator.get("initial_risk_level"),
        "aml_auditor": aml,
        "aml_decision": aml.get("decision"),
        "matched_typology": aml.get("matched_typology"),
        "aml_confidence": aml.get("confidence"),
        "aml_reasoning": aml.get("reasoning"),
        "regulatory_findings": regulatory,
        "consolidated_evidence": evidence.get("consolidated_evidence", []),
        "evidence_sources": evidence.get("evidence_sources", {}),
        "sar_draft": draft,
        "narrative": draft.get("narrative"),
        "structured_fields": draft.get("structured_fields", {}),
        "draft_version": draft.get("draft_version"),
        "compliance_check": state.get("compliance_check"),
        "trace": state.get("trace", []),
        "status": state.get("status", "unknown"),
        "screening_id": _case_metadata(state["transaction_id"]).get("screening_id"),
    }


def _screen_dataframe(data: bytes) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    try:
        raw = pd.read_csv(io.BytesIO(data))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid CSV file: {exc}") from exc
    if raw.empty:
        raise HTTPException(status_code=400, detail="The uploaded CSV is empty.")
    missing = sorted(REQUIRED_COLUMNS - set(raw.columns))
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing required columns: {', '.join(missing)}")

    df = raw.copy()
    try:
        df = df.dropna().reset_index(drop=True)
        if df.empty:
            raise HTTPException(status_code=400, detail="No complete transactions remain after removing missing values.")
        original_dates = df["Date"].astype(str).copy()
        original_times = df["Time"].astype(str).copy()
        df["Date"] = pd.to_datetime(df["Date"])
        df["Year"] = df["Date"].dt.year
        df["Month"] = df["Date"].dt.month
        df["Day"] = df["Date"].dt.day
        df["Week"] = df["Date"].dt.isocalendar().week.astype(int)

        sender_stats = df.groupby("Sender_account")["Amount"].agg(["count", "mean", "std"]).rename(
            columns={"count": "Sender_txn_count", "mean": "Sender_avg_amount", "std": "Sender_std_amount"}
        )
        df = df.merge(sender_stats, on="Sender_account", how="left")
        df["Sender_std_amount"] = df["Sender_std_amount"].fillna(0)

        receiver_stats = df.groupby("Receiver_account")["Amount"].agg(["count", "mean"]).rename(
            columns={"count": "Receiver_txn_count", "mean": "Receiver_avg_amount"}
        )
        df = df.merge(receiver_stats, on="Receiver_account", how="left")
        df["Amount_dev_from_sender_avg"] = (
            (df["Amount"] - df["Sender_avg_amount"]) / (df["Sender_std_amount"] + 1)
        )
        df["Is_cross_border"] = (df["Sender_bank_location"] != df["Receiver_bank_location"]).astype(int)
        df["Is_round_amount"] = (df["Amount"] % 1000 == 0).astype(int)
        df["Near_reporting_threshold"] = ((df["Amount"] >= 9000) & (df["Amount"] < 10000)).astype(int)

        model_input = df.drop(columns=["Sender_account", "Receiver_account", "Time", "Date"])
        for column in _model_categorical_columns:
            model_input[column] = pd.Categorical(model_input[column], categories=_category_mappings[column])
        model_input = model_input[_model_feature_columns]
        probabilities = _model.predict_proba(model_input)[:, 1]
        df["risk_probability"] = probabilities
        df["sar_worthy"] = (probabilities >= SCREENING_THRESHOLD).astype(int)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Unable to screen uploaded transactions: {exc}") from exc

    results: list[dict[str, Any]] = []
    for index, row in df.iterrows():
        results.append(
            {
                "transaction_id": str(index),
                "date": original_dates.iloc[index],
                "time": original_times.iloc[index],
                "sender_account": _json_value(row["Sender_account"]),
                "receiver_account": _json_value(row["Receiver_account"]),
                "amount": _json_value(row["Amount"]),
                "payment_currency": _json_value(row["Payment_currency"]),
                "received_currency": _json_value(row["Received_currency"]),
                "payment_type": _json_value(row["Payment_type"]),
                "sender_bank_location": _json_value(row["Sender_bank_location"]),
                "receiver_bank_location": _json_value(row["Receiver_bank_location"]),
                "risk_probability": _json_value(row["risk_probability"]),
                "sar_worthy": int(row["sar_worthy"]),
                "sender_txn_count": _json_value(row["Sender_txn_count"]),
                "receiver_txn_count": _json_value(row["Receiver_txn_count"]),
                "amount_dev_from_sender_avg": _json_value(row["Amount_dev_from_sender_avg"]),
                "is_cross_border": bool(row["Is_cross_border"]),
                "is_round_amount": bool(row["Is_round_amount"]),
                "near_reporting_threshold": bool(row["Near_reporting_threshold"]),
            }
        )
    return df, results


def _alert_from_screening(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "transaction_id": str(record["transaction_id"]),
        "sender_account": str(record["sender_account"]),
        "receiver_account": str(record["receiver_account"]),
        "amount": float(record["amount"]),
        "payment_type": str(record["payment_type"]),
        "payment_currency": str(record["payment_currency"]),
        "received_currency": str(record["received_currency"]),
        "sender_bank_location": str(record["sender_bank_location"]),
        "receiver_bank_location": str(record["receiver_bank_location"]),
        "transaction_date": str(record["date"]),
        "transaction_time": str(record["time"]),
        "xgb_score": float(record["risk_probability"]),
        "xgb_top_features": [],
        "sender_txn_count": int(record["sender_txn_count"]),
        "receiver_txn_count": int(record["receiver_txn_count"]),
        "amount_dev_from_sender_avg": float(record["amount_dev_from_sender_avg"]),
        "is_cross_border": bool(record["is_cross_border"]),
        "is_round_amount": bool(record["is_round_amount"]),
        "near_reporting_threshold": bool(record["near_reporting_threshold"]),
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/screen")
async def screen(file: UploadFile = File(...)) -> dict[str, Any]:
    filename = file.filename or ""
    if not filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="The uploaded file must be a CSV.")
    data = await file.read()
    async with _screen_lock:
        dataframe, results = _screen_dataframe(data)
        screening_id = uuid.uuid4().hex
        _screenings[screening_id] = {
            "screening_id": screening_id,
            "dataframe": dataframe,
            "results": results,
            "created_at": datetime.utcnow().isoformat(),
        }
    return {
        "screening_id": screening_id,
        "total_transactions": len(results),
        "suspicious_count": sum(item["sar_worthy"] == 1 for item in results),
        "results": results,
        "memory_saver_note": "Screening data is process-local and is lost when the API restarts.",
    }


@app.get("/screening/{screening_id}")
def get_screening(screening_id: str) -> dict[str, Any]:
    screening = _screenings.get(screening_id)
    if screening is None:
        raise HTTPException(status_code=404, detail="Screening result not found in this API process.")
    results = screening["results"]
    return {
        "screening_id": screening_id,
        "total_transactions": len(results),
        "suspicious_count": sum(item["sar_worthy"] == 1 for item in results),
        "results": results,
    }


@app.get("/screening/{screening_id}/csv")
def download_screening_csv(screening_id: str) -> StreamingResponse:
    screening = _screenings.get(screening_id)
    if screening is None:
        raise HTTPException(status_code=404, detail="Screening result not found in this API process.")
    output = io.StringIO()
    screening["dataframe"].to_csv(output, index=False)
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="screening-{screening_id}.csv"'},
    )


async def _run_case(screening_id: str, record: dict[str, Any], semaphore: asyncio.Semaphore) -> dict[str, Any]:
    transaction_id = str(record["transaction_id"])
    async with semaphore:
        try:
            result = await _graph.ainvoke(
                create_initial_state(_alert_from_screening(record)),
                config=_config(transaction_id),
            )
            _cases[transaction_id] = {"screening_id": screening_id}
            state = _state_for(transaction_id) or dict(result)
            state["transaction_id"] = transaction_id
            if "__interrupt__" in result:
                state["status"] = "awaiting_human_review"
            return {"transaction_id": transaction_id, "status": state.get("status", "unknown")}
        except Exception as exc:
            _cases[transaction_id] = {"screening_id": screening_id, "error": str(exc)}
            return {"transaction_id": transaction_id, "status": "error", "detail": str(exc)}


@app.post("/cases/run")
async def run_cases(request: RunCasesRequest) -> dict[str, Any]:
    screening = _screenings.get(request.screening_id)
    if screening is None:
        raise HTTPException(status_code=404, detail="Screening result not found in this API process.")
    records = {str(record["transaction_id"]): record for record in screening["results"]}
    unknown = [transaction_id for transaction_id in request.transaction_ids if transaction_id not in records]
    if unknown:
        raise HTTPException(status_code=404, detail=f"Transaction IDs not found in screening: {unknown}")
    unselected_non_suspicious = [
        transaction_id for transaction_id in request.transaction_ids if records[transaction_id]["sar_worthy"] != 1
    ]
    if unselected_non_suspicious:
        raise HTTPException(
            status_code=400,
            detail=f"Only suspicious transactions can be run: {unselected_non_suspicious}",
        )

    async with _case_run_lock:
        semaphore = asyncio.Semaphore(MAX_CASE_CONCURRENCY)
        outcomes = await asyncio.gather(
            *[_run_case(request.screening_id, records[transaction_id], semaphore) for transaction_id in request.transaction_ids]
        )
    return {
        "screening_id": request.screening_id,
        "started": [item["transaction_id"] for item in outcomes if item["status"] != "error"],
        "awaiting_human_review": [item["transaction_id"] for item in outcomes if item["status"] == "awaiting_human_review"],
        "errors": [item for item in outcomes if item["status"] == "error"],
        "results": outcomes,
    }


@app.get("/cases")
def list_cases() -> dict[str, Any]:
    cases = []
    for transaction_id in sorted(_cases):
        if "error" in _cases[transaction_id]:
            cases.append({"transaction_id": transaction_id, "status": "error", **_cases[transaction_id]})
            continue
        state = _state_for(transaction_id)
        if state is not None:
            cases.append(_status_summary(transaction_id, state))
    return {"cases": cases}


@app.get("/cases/{transaction_id}")
def get_case(transaction_id: str) -> dict[str, Any]:
    return _case_response(_require_case(transaction_id))


@app.get("/cases/{transaction_id}/trace")
def get_case_trace(transaction_id: str) -> dict[str, Any]:
    state = _require_case(transaction_id)
    return {
        "transaction_id": transaction_id,
        "trace": state.get("trace", []),
    }


def _resume(transaction_id: str, payload: dict[str, str]) -> dict[str, Any]:
    state = _require_case(transaction_id)
    if state.get("status") != "awaiting_human_review":
        raise HTTPException(status_code=409, detail="Case is not awaiting human review.")
    try:
        result = _graph.invoke(Command(resume=payload), config=_config(transaction_id))
        updated = _state_for(transaction_id) or dict(result)
        updated["transaction_id"] = transaction_id
        return _case_response(updated)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Unable to resume workflow: {exc}") from exc


@app.post("/cases/{transaction_id}/approve")
def approve_case(transaction_id: str) -> dict[str, Any]:
    return _resume(transaction_id, {"decision": "approve"})


@app.post("/cases/{transaction_id}/reject")
def reject_case(transaction_id: str, request: ReasonRequest) -> dict[str, Any]:
    return _resume(transaction_id, {"decision": "reject", "reason": request.reason})


@app.post("/cases/{transaction_id}/revise")
def revise_case(transaction_id: str, request: ReasonRequest) -> dict[str, Any]:
    return _resume(transaction_id, {"decision": "revise", "reason": request.reason})


def _pdf_lines(state: dict[str, Any]) -> list[str]:
    report = _case_response(state)
    lines = [
        "AML SUSPICIOUS ACTIVITY REPORT",
        "",
        "CASE INFORMATION",
        f"Transaction ID: {report['transaction_id']}",
        f"Final Status: {report['status']}",
        "",
        "TRANSACTION DETAILS",
    ]
    for key in (
        "transaction_date", "transaction_time", "sender_account", "receiver_account", "amount",
        "payment_type", "payment_currency", "received_currency", "sender_bank_location", "receiver_bank_location",
    ):
        lines.append(f"{key.replace('_', ' ').title()}: {report.get(key)}")
    lines.extend(["", f"XGBoost Risk Score: {report.get('xgb_score')}", "", "INVESTIGATOR FINDINGS"])
    lines.extend([f"Anomaly Summary: {report.get('anomaly_summary')}", f"Flagged Signals: {report.get('flagged_signals')}", f"Initial Risk Level: {report.get('initial_risk_level')}"])
    lines.extend(["", "AML AUDITOR DECISION", str(report.get("aml_auditor")), "", "REGULATORY FINDINGS", str(report.get("regulatory_findings")), "", "EVIDENCE SUMMARY"])
    lines.extend([str(report.get("consolidated_evidence")), f"Evidence Sources: {report.get('evidence_sources')}", "", "SAR STRUCTURED FIELDS", str(report.get("structured_fields")), "", "FULL SAR NARRATIVE", str(report.get("narrative")), "", "COMPLIANCE AUDITOR", str(report.get("compliance_check")), "", "COMPLETE AGENT TRACE"])
    for entry in report.get("trace", []):
        lines.extend(
            [
                f"Agent: {entry.get('agent')}",
                f"Action: {entry.get('action')}",
                f"Input: {entry.get('input_summary')}",
                f"Output: {entry.get('output_summary')}",
                f"Metadata: {entry.get('metadata')}",
                f"Timestamp: {entry.get('timestamp')}",
                f"Duration: {entry.get('duration_ms')} ms",
                "",
            ]
        )
    return lines


@app.get("/cases/{transaction_id}/pdf")
def download_case_pdf(transaction_id: str) -> Response:
    state = _require_case(transaction_id)
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.lib.units import inch
        from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer
    except ImportError as exc:
        raise HTTPException(status_code=500, detail="PDF support is unavailable; install reportlab.") from exc

    buffer = io.BytesIO()
    document = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=0.6 * inch, leftMargin=0.6 * inch, topMargin=0.6 * inch, bottomMargin=0.6 * inch)
    styles = getSampleStyleSheet()
    story = []
    for line in _pdf_lines(state):
        if not line:
            story.append(Spacer(1, 8))
        else:
            story.append(Paragraph(str(line).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"), styles["BodyText"]))
            story.append(Spacer(1, 4))
    document.build(story)
    return Response(
        content=buffer.getvalue(),
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="sar-{transaction_id}.pdf"'},
    )
