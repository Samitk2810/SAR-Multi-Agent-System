"""Modern Streamlit frontend for the AML/SAR FastAPI backend."""

from __future__ import annotations

from typing import Any

import pandas as pd
import requests
import streamlit as st


API_BASE_URL = "http://127.0.0.1:8000"
REQUEST_TIMEOUT = 300

st.set_page_config(page_title="AML SAR System", page_icon="", layout="wide", initial_sidebar_state="expanded")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
    :root { --ink:#132127; --muted:#708087; --line:#dce6e5; --mint:#c9f2e6; --teal:#0f766e; --amber:#f4b942; --paper:#f7faf9; --text-color:var(--ink); --background-color:var(--paper); --secondary-background-color:#fff; --primary-color:var(--teal); }
    html, body, [class*="css"] { font-family:'DM Sans', sans-serif; color:var(--ink); }
    [data-testid="stAppViewContainer"] { background:var(--paper); color:var(--ink); }
    [data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"],
    [data-testid="stAppViewContainer"] [data-testid="stWidgetLabel"] *,
    [data-testid="stAppViewContainer"] [data-testid="stCaptionContainer"] * { color:var(--ink); }
    [data-testid="stHeader"] { background:rgba(247,250,249,.88); }
    [data-testid="stSidebar"] { background:#10272b; border-right:0; }
    [data-testid="stSidebar"] * { color:#eaf7f3 !important; }
    [data-testid="stSidebar"] .stRadio label { padding:8px 10px; border-radius:10px; }
    [data-testid="stSidebar"] .stRadio label:hover { background:rgba(201,242,230,.12); }
    h1, h2, h3 { font-family:'Space Grotesk', sans-serif; letter-spacing:-.025em; }
    h1 { font-size:2.35rem !important; margin-bottom:.2rem !important; }
    h2 { font-size:1.35rem !important; margin-top:1.5rem !important; }
    h3 { font-size:1.05rem !important; }
    .eyebrow { color:var(--teal); font-size:.72rem; font-weight:700; letter-spacing:.14em; text-transform:uppercase; }
    .lede { color:var(--muted); font-size:1rem; margin:0 0 1.25rem; }
    .hero { background:linear-gradient(120deg,#e4f7f0 0%,#f7faf9 58%,#fff7e7 100%); border:1px solid #d5e9e4; border-radius:22px; padding:30px 34px; margin:8px 0 24px; }
    .metric-band { background:white; border:1px solid var(--line); border-radius:16px; padding:16px 18px; min-height:102px; box-shadow:0 7px 20px rgba(19,33,39,.04); }
    .metric-label { color:var(--muted); font-size:.76rem; text-transform:uppercase; letter-spacing:.08em; }
    .metric-value { color:var(--ink); font-family:'Space Grotesk'; font-size:1.42rem; font-weight:700; margin-top:8px; }
    .section-kicker { border-bottom:1px solid var(--line); padding-bottom:9px; margin-top:26px; }
    .status-pill { display:inline-block; border-radius:999px; padding:5px 10px; font-size:.75rem; font-weight:700; background:#e7f5f1; color:#0f766e; }
    .status-pill.waiting { background:#fff1cf; color:#9a6500; }
    .status-pill.error { background:#fee7e5; color:#aa3029; }
    .trace-card { background:white; border:1px solid var(--line); border-left:4px solid var(--teal); border-radius:14px; padding:14px 18px; margin:10px 0; box-shadow:0 5px 16px rgba(19,33,39,.035); }
    .trace-index { color:var(--teal); font-family:'Space Grotesk'; font-size:.78rem; font-weight:700; }
    .trace-agent { font-family:'Space Grotesk'; font-size:1.05rem; font-weight:700; margin:2px 0 3px; }
    .trace-action { color:var(--muted); font-size:.9rem; }
    .upload-zone { background:white; border:1px dashed #9fc9c0; border-radius:16px; padding:5px; }
    .small-note { color:var(--muted); font-size:.82rem; }
    div[data-testid="stDataFrame"] { border:1px solid var(--line); border-radius:14px; overflow:hidden; }
    .stButton > button, .stDownloadButton > button, [data-testid="stFileUploader"] button { border-radius:10px; font-weight:700; background:var(--teal) !important; border:1px solid var(--teal) !important; color:#fff !important; }
    .stButton > button *, .stDownloadButton > button *, [data-testid="stFileUploader"] button * { color:#fff !important; }
    div[data-testid="stExpander"] details > summary { background:#e9eef0 !important; color:var(--ink) !important; }
    div[data-testid="stExpander"] details > summary * { color:var(--ink) !important; fill:var(--ink) !important; }
    </style>
    """,
    unsafe_allow_html=True,
)


def api_request(method: str, endpoint: str, **kwargs: Any) -> tuple[Any | None, str | None]:
    try:
        response = requests.request(method, f"{API_BASE_URL}{endpoint}", timeout=REQUEST_TIMEOUT, **kwargs)
    except requests.exceptions.Timeout:
        return None, "The API request timed out. Confirm FastAPI is running."
    except requests.exceptions.ConnectionError:
        return None, f"API unavailable at {API_BASE_URL}. Start FastAPI and try again."
    except requests.exceptions.RequestException:
        return None, "The API request could not be completed."
    try:
        payload = response.json()
    except ValueError:
        return None, f"The API returned an invalid response (HTTP {response.status_code})."
    if not response.ok:
        detail = payload.get("detail", "The API rejected the request.") if isinstance(payload, dict) else str(payload)
        return None, f"{detail} (HTTP {response.status_code})"
    return payload, None


def download_request(endpoint: str) -> tuple[bytes | None, str | None]:
    try:
        response = requests.get(f"{API_BASE_URL}{endpoint}", timeout=REQUEST_TIMEOUT)
    except requests.exceptions.Timeout:
        return None, "The download request timed out."
    except requests.exceptions.ConnectionError:
        return None, f"API unavailable at {API_BASE_URL}."
    except requests.exceptions.RequestException:
        return None, "The download request could not be completed."
    if not response.ok:
        try:
            detail = response.json().get("detail", "Download failed.")
        except ValueError:
            detail = "Download failed."
        return None, f"{detail} (HTTP {response.status_code})"
    return response.content, None


def error(message: str) -> None:
    if "rate limit" in message.lower() or "free-models" in message.lower():
        st.error(f"The configured LLM provider has reached its current rate limit. {message}")
    else:
        st.error(message)


def text(value: Any, fallback: str = "Not available") -> str:
    if value is None or value == "":
        return fallback
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    return str(value)


def status_pill(status: str) -> None:
    style = "waiting" if status == "awaiting_human_review" else "error" if status == "error" else ""
    st.markdown(f'<span class="status-pill {style}">{status.replace("_", " ").upper()}</span>', unsafe_allow_html=True)


def init_state() -> None:
    defaults = {
        "page": "Screening & Cases",
        "screening_id": None,
        "screening_result": None,
        "screening_file_name": None,
        "selected_ids": [],
        "cases": [],
        "case_details": {},
        "traces": {},
        "selected_case": None,
        "last_run": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def refresh_cases() -> None:
    payload, message = api_request("GET", "/cases")
    if message:
        error(message)
        return
    st.session_state.cases = payload.get("cases", [])
    st.session_state.case_details = {}
    st.session_state.traces = {}


def get_case(transaction_id: str) -> dict[str, Any] | None:
    if transaction_id not in st.session_state.case_details:
        payload, message = api_request("GET", f"/cases/{transaction_id}")
        if message:
            error(message)
            return None
        st.session_state.case_details[transaction_id] = payload
    return st.session_state.case_details[transaction_id]


def get_trace(transaction_id: str) -> list[dict[str, Any]] | None:
    if transaction_id not in st.session_state.traces:
        payload, message = api_request("GET", f"/cases/{transaction_id}/trace")
        if message:
            error(message)
            return None
        st.session_state.traces[transaction_id] = payload.get("trace", [])
    return st.session_state.traces[transaction_id]


def metric(label: str, value: Any) -> None:
    st.markdown(
        f'<div class="metric-band"><div class="metric-label">{label}</div><div class="metric-value">{text(value)}</div></div>',
        unsafe_allow_html=True,
    )


def render_header(eyebrow: str, title: str, subtitle: str) -> None:
    st.markdown(f'<div class="eyebrow">{eyebrow}</div>', unsafe_allow_html=True)
    st.title(title)
    st.markdown(f'<p class="lede">{subtitle}</p>', unsafe_allow_html=True)


def screen_page() -> None:
    render_header("CONTROL ROOM / 01", "AML Screening & Case Management", "Upload a transaction batch, inspect model decisions, and route only the cases you choose into multi-agent analysis.")

    st.markdown('<div class="hero"><div class="eyebrow">SCREENING INTAKE</div><h3>Turn a raw transaction file into a review queue</h3><p class="small-note">The FastAPI backend runs the existing XGBoost pipeline. Nothing is scored or fabricated in this interface.</p></div>', unsafe_allow_html=True)
    upload = st.file_uploader("Transaction CSV", type=["csv"], label_visibility="collapsed", key="screen_upload")
    if upload:
        st.caption(f"Ready to process · {upload.name} · {upload.size:,} bytes")
    if st.button("Run XGBoost Screening", type="primary", disabled=upload is None, width='content'):
        with st.spinner("Sending transaction batch to the screening service..."):
            payload, message = api_request("POST", "/screen", files={"file": (upload.name, upload.getvalue(), "text/csv")})
        if message:
            error(message)
        else:
            st.session_state.screening_id = payload.get("screening_id")
            st.session_state.screening_result = payload
            st.session_state.screening_file_name = upload.name
            st.session_state.selected_ids = []
            refresh_cases()
            st.success("Screening completed. Review the full result set below.")

    result = st.session_state.screening_result
    if not result:
        st.info("Upload a CSV to begin. Your screening result will remain available while you move between pages.")
        return

    results = result.get("results", [])
    suspicious = [row for row in results if row.get("sar_worthy") == 1]
    summary = st.columns(4)
    for col, label, value in zip(summary, ["Screening ID", "Transactions", "Suspicious", "Non-suspicious"], [result.get("screening_id"), len(results), len(suspicious), len(results) - len(suspicious)]):
        with col:
            metric(label, value)

    st.markdown('<div class="section-kicker"><div class="eyebrow">MODEL OUTPUT</div><h2>Screening Results</h2></div>', unsafe_allow_html=True)
    table_rows = []
    for row in results:
        table_rows.append({
            "Status": "SUSPICIOUS" if row.get("sar_worthy") == 1 else "CLEAR",
            "Transaction ID": row.get("transaction_id"),
            "Date": row.get("date"),
            "Time": row.get("time"),
            "Sender": row.get("sender_account"),
            "Receiver": row.get("receiver_account"),
            "Amount": row.get("amount"),
            "Currency": row.get("payment_currency"),
            "Payment": row.get("payment_type"),
            "Risk probability": row.get("risk_probability"),
            "SAR worthy": row.get("sar_worthy"),
        })
    st.dataframe(pd.DataFrame(table_rows), width='stretch', hide_index=True)
    csv_bytes, csv_message = download_request(f"/screening/{result.get('screening_id')}/csv")
    if csv_message:
        st.caption(f"Screening CSV download unavailable: {csv_message}")
    else:
        st.download_button("Download Screening CSV", csv_bytes, file_name=f"screening-{result.get('screening_id')}.csv", mime="text/csv")

    st.markdown('<div class="section-kicker"><div class="eyebrow">CASE ROUTING</div><h2>Select Suspicious Cases</h2></div>', unsafe_allow_html=True)
    if not suspicious:
        st.warning("No suspicious transactions were returned by the model.")
        return
    all_ids = [str(row["transaction_id"]) for row in suspicious]
    select_all = st.checkbox("Select All Suspicious Cases", value=set(st.session_state.selected_ids) == set(all_ids), key="select_all_cases")
    selected = all_ids if select_all else []
    if not select_all:
        selected = [transaction_id for transaction_id in all_ids if st.checkbox(f"Transaction {transaction_id}", value=transaction_id in st.session_state.selected_ids, key=f"screen_case_{transaction_id}")]
    st.session_state.selected_ids = selected
    st.markdown(f"**{len(selected)} cases selected**")
    if st.button("Run Multi-Agent Analysis", type="primary", disabled=not selected):
        with st.spinner(f"Running the existing SAR workflow for {len(selected)} selected case(s)..."):
            payload, message = api_request("POST", "/cases/run", json={"screening_id": result.get("screening_id"), "transaction_ids": selected})
        if message:
            error(message)
        else:
            st.session_state.last_run = payload
            refresh_cases()
            st.success("Selected cases have been submitted to the SAR workflow.")

    if st.session_state.last_run:
        st.markdown('<div class="section-kicker"><div class="eyebrow">RUN STATUS</div><h2>Latest Multi-Agent Run</h2></div>', unsafe_allow_html=True)
        for item in st.session_state.last_run.get("results", []):
            cols = st.columns([2, 2, 5])
            cols[0].markdown(f"**Transaction {item.get('transaction_id')}**")
            cols[1].markdown(f"`{item.get('status', 'unknown').replace('_', ' ')}`")
            if item.get("detail"):
                cols[2].warning(item["detail"])
            else:
                cols[2].caption("State captured by the existing LangGraph execution.")


def render_mapping(mapping: Any) -> None:
    if not isinstance(mapping, dict) or not mapping:
        st.caption("No information returned.")
        return
    values = list(mapping.items())
    cols = st.columns(2)
    for index, (key, value) in enumerate(values):
        cols[index % 2].markdown(f"**{key.replace('_', ' ').title()}**  \n{text(value)}")


def case_report(case: dict[str, Any]) -> None:
    transaction_id = str(case.get("transaction_id"))
    status = text(case.get("status"), "unknown")
    render_header("CASE REPORT / REVIEW", f"Transaction {transaction_id}", "A complete report assembled from the existing screening and LangGraph state.")
    top = st.columns(4)
    for col, label, value in zip(top, ["Current status", "XGBoost score", "AML decision", "Draft version"], [status.replace("_", " "), case.get("xgb_score"), case.get("aml_decision"), case.get("draft_version")]):
        with col:
            metric(label, value)
    status_pill(status)

    with st.expander("Transaction Details", expanded=True):
        details = [("Transaction date", "transaction_date"), ("Transaction time", "transaction_time"), ("Sender", "sender_account"), ("Receiver", "receiver_account"), ("Amount", "amount"), ("Payment type", "payment_type"), ("Payment currency", "payment_currency"), ("Received currency", "received_currency"), ("Sender bank", "sender_bank_location"), ("Receiver bank", "receiver_bank_location")]
        cols = st.columns(4)
        for index, (label, key) in enumerate(details):
            cols[index % 4].markdown(f"**{label}**  \n{text(case.get(key))}")
    with st.expander("Investigator Findings", expanded=True):
        st.markdown(f"**Anomaly summary**  \n{text(case.get('anomaly_summary'))}")
        st.markdown(f"**Flagged signals**  \n{text(case.get('flagged_signals'))}")
        st.markdown(f"**Initial risk level**  \n{text(case.get('initial_risk_level'))}")
    with st.expander("AML Auditor Decision", expanded=True):
        st.markdown(f"**Matched typology:** {text(case.get('matched_typology'))}  ·  **Confidence:** {text(case.get('aml_confidence'))}")
        st.markdown(f"**Reasoning**  \n{text(case.get('aml_reasoning'))}")
    with st.expander("Regulatory Findings"):
        render_mapping(case.get("regulatory_findings"))
    with st.expander("Evidence"):
        evidence = case.get("consolidated_evidence") or []
        for item in evidence:
            st.markdown(f"- {item}")
        st.markdown("**Sources**")
        render_mapping(case.get("evidence_sources"))
    with st.expander("SAR Report", expanded=True):
        st.markdown(f"**Draft version:** {text(case.get('draft_version'))}")
        st.markdown("**Structured fields**")
        render_mapping(case.get("structured_fields"))
        st.markdown("**SAR Narrative**")
        st.text_area("SAR Narrative", text(case.get("narrative")), height=330, disabled=True, label_visibility="collapsed")
    with st.expander("Compliance Review", expanded=True):
        compliance = case.get("compliance_check") or {}
        if compliance.get("result") == "pass":
            st.success("PASS")
        else:
            st.warning(text(compliance.get("result")).upper())
        for issue in compliance.get("issues") or []:
            st.markdown(f"- {issue}")

    pdf_bytes, pdf_message = download_request(f"/cases/{transaction_id}/pdf")
    if pdf_message:
        st.warning(f"PDF unavailable: {pdf_message}")
    else:
        st.download_button("Download SAR PDF", pdf_bytes, file_name=f"sar-{transaction_id}.pdf", mime="application/pdf")

    if status == "awaiting_human_review":
        st.markdown('<div class="section-kicker"><div class="eyebrow">DECISION GATE</div><h2>Human Review Required</h2></div>', unsafe_allow_html=True)
        actions = st.columns(3)
        with actions[0]:
            if st.button("Approve", type="primary", width='stretch'):
                payload, message = api_request("POST", f"/cases/{transaction_id}/approve")
                if message:
                    error(message)
                else:
                    st.session_state.case_details.pop(transaction_id, None)
                    refresh_cases()
                    st.success(f"Case filed. Status: {text(payload.get('status'))}")
        with actions[1]:
            reason = st.text_area("Rejection reason", key=f"reject_reason_{transaction_id}")
            if st.button("Reject", width='stretch'):
                if not reason.strip():
                    st.error("A rejection reason is required.")
                else:
                    payload, message = api_request("POST", f"/cases/{transaction_id}/reject", json={"reason": reason.strip()})
                    if message:
                        error(message)
                    else:
                        st.session_state.case_details.pop(transaction_id, None)
                        refresh_cases()
                        st.warning(f"Case dismissed. Status: {text(payload.get('status'))}")
        with actions[2]:
            reason = st.text_area("Revision reason", key=f"revise_reason_{transaction_id}")
            if st.button("Request Revision", width='stretch'):
                if not reason.strip():
                    st.error("A revision reason is required.")
                else:
                    payload, message = api_request("POST", f"/cases/{transaction_id}/revise", json={"reason": reason.strip()})
                    if message:
                        error(message)
                    else:
                        st.session_state.case_details.pop(transaction_id, None)
                        refresh_cases()
                        st.success(f"Sent back to SAR Drafter. Status: {text(payload.get('status'))}")


def reports_page() -> None:
    render_header("CONTROL ROOM / 02", "Case Reports", "Review analyzed cases, read the SAR narrative, and make the human disposition when required.")
    if not st.session_state.cases:
        refresh_cases()
    case_ids = [str(item.get("transaction_id")) for item in st.session_state.cases]
    if not case_ids:
        st.info("No analyzed cases are available yet. Run selected cases from Screening & Cases.")
        return
    selected = st.selectbox("Select case", case_ids, index=case_ids.index(st.session_state.selected_case) if st.session_state.selected_case in case_ids else 0, key="report_case_selector")
    st.session_state.selected_case = selected
    case = get_case(selected)
    if case:
        case_report(case)


def total_duration(trace: list[dict[str, Any]]) -> float | None:
    durations = [entry.get("duration_ms") for entry in trace if isinstance(entry.get("duration_ms"), (int, float))]
    return sum(durations) if durations else None


def trace_page() -> None:
    render_header("OBSERVABILITY / 03", "Agent Execution Trace", "An ordered audit timeline from the actual LangGraph trace. Every entry below is returned by the backend.")
    if not st.session_state.cases:
        refresh_cases()
    case_ids = [str(item.get("transaction_id")) for item in st.session_state.cases]
    if not case_ids:
        st.info("No analyzed cases are available yet.")
        return
    selected = st.selectbox("Select case", case_ids, index=case_ids.index(st.session_state.selected_case) if st.session_state.selected_case in case_ids else 0, key="trace_case_selector")
    st.session_state.selected_case = selected
    trace = get_trace(selected)
    case = get_case(selected)
    if trace is None or case is None:
        return
    duration = total_duration(trace)
    summary = st.columns(3)
    for col, label, value in zip(summary, ["Transaction ID", "Current status", "Trace entries"], [selected, case.get("status"), len(trace)]):
        with col:
            metric(label, value)
    if duration is not None:
        st.caption(f"Total recorded workflow duration · {duration:.2f} ms")
    st.markdown('<div class="section-kicker"><div class="eyebrow">EXECUTION TIMELINE</div><h2>Agent Execution Trace</h2></div>', unsafe_allow_html=True)
    if not trace:
        st.info("No trace entries were returned for this case.")
        return
    for index, entry in enumerate(trace, start=1):
        st.markdown('<div class="trace-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="trace-index">STEP {index:02d}</div><div class="trace-agent">{text(entry.get("agent"))}</div><div class="trace-action">{text(entry.get("action"))}</div>', unsafe_allow_html=True)
        info = st.columns(2)
        info[0].caption(f"Timestamp · {text(entry.get('timestamp'))}")
        info[1].caption(f"Duration · {text(entry.get('duration_ms'))} ms")
        st.markdown('</div>', unsafe_allow_html=True)
        with st.expander(f"Inspect step {index:02d}"):
            st.markdown(f"**Input**  \n{text(entry.get('input_summary'))}")
            st.markdown(f"**Output**  \n{text(entry.get('output_summary'))}")
            st.markdown(f"**Metadata**  \n{text(entry.get('metadata'))}")


init_state()

with st.sidebar:
    st.markdown("<div class='eyebrow'>AML SAR SYSTEM</div>", unsafe_allow_html=True)
    st.markdown("## Compliance workspace")
    health, health_error = api_request("GET", "/health")
    if health_error:
        st.error("API unavailable")
    else:
        st.success("API connected")
    st.divider()
    st.session_state.page = st.radio("Navigate", ["Screening & Cases", "Case Reports", "Agent Execution Trace"], index=["Screening & Cases", "Case Reports", "Agent Execution Trace"].index(st.session_state.page), label_visibility="collapsed")
    st.divider()
    st.caption("SOURCE OF TRUTH")
    st.caption("All screening, SAR state, decisions, and traces are served by FastAPI.")
    if st.button("Refresh API state", width='stretch'):
        refresh_cases()
        st.rerun()

if st.session_state.page == "Screening & Cases":
    screen_page()
elif st.session_state.page == "Case Reports":
    reports_page()
else:
    trace_page()
