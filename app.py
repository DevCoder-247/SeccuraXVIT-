"""
Autonomous Document Verification System
Built with Streamlit + LangGraph + Gemini 2.5
"""

import os
import sys
import base64
import json
from datetime import datetime
from pathlib import Path
from io import BytesIO

import streamlit as st
from PIL import Image
from dotenv import load_dotenv

# ─── Path setup ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
load_dotenv()

# ─── Page config (must be first Streamlit call) ───────────────────────────────
st.set_page_config(
    page_title="DocVerify AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --verified: #00c896;
    --verified-bg: #001f17;
    --invalid: #ff4757;
    --invalid-bg: #1f0007;
    --unverifiable: #ffb300;
    --unverifiable-bg: #1f1500;
    --human-approved: #00b4d8;
    --human-rejected: #e63946;
    --bg-deep: #080c14;
    --bg-card: #0d1320;
    --bg-card2: #111827;
    --border: #1e2d45;
    --text: #e2e8f0;
    --text-dim: #64748b;
    --accent: #3b82f6;
    --accent2: #6366f1;
}

html, body, .stApp {
    background-color: var(--bg-deep) !important;
    color: var(--text) !important;
    font-family: 'Space Grotesk', sans-serif !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--bg-card) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { font-family: 'Space Grotesk', sans-serif !important; }

/* ── Headings ── */
h1, h2, h3, h4, h5 {
    font-family: 'Space Grotesk', sans-serif !important;
    color: var(--text) !important;
    font-weight: 700 !important;
    letter-spacing: -0.02em !important;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, var(--accent), var(--accent2)) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 600 !important;
    letter-spacing: 0.03em !important;
    padding: 0.6rem 1.5rem !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 20px rgba(59,130,246,0.4) !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: var(--bg-card) !important;
    border-bottom: 1px solid var(--border) !important;
    gap: 0 !important;
}
.stTabs [data-baseweb="tab"] {
    color: var(--text-dim) !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 500 !important;
    padding: 0.75rem 1.5rem !important;
}
.stTabs [aria-selected="true"] {
    color: var(--accent) !important;
    background: transparent !important;
    border-bottom: 2px solid var(--accent) !important;
}

/* ── Cards ── */
.card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 1rem;
}
.card-header {
    font-size: 0.7rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--text-dim);
    margin-bottom: 0.5rem;
}

/* ── Verdict banner ── */
.verdict-approved {
    background: linear-gradient(135deg, #001f17, #002b1f);
    border: 1px solid var(--verified);
    border-radius: 12px;
    padding: 1.5rem 2rem;
    text-align: center;
    margin: 1rem 0;
}
.verdict-review {
    background: linear-gradient(135deg, #1f1500, #2b1e00);
    border: 1px solid var(--unverifiable);
    border-radius: 12px;
    padding: 1.5rem 2rem;
    text-align: center;
    margin: 1rem 0;
}
.verdict-rejected {
    background: linear-gradient(135deg, #1f0007, #2b000d);
    border: 1px solid var(--invalid);
    border-radius: 12px;
    padding: 1.5rem 2rem;
    text-align: center;
    margin: 1rem 0;
}

/* ── Field result rows ── */
.field-row {
    display: flex;
    align-items: flex-start;
    gap: 1rem;
    padding: 0.8rem 1rem;
    border-radius: 8px;
    margin: 0.4rem 0;
    border: 1px solid transparent;
    font-family: 'Space Grotesk', sans-serif;
}
.field-verified { background: var(--verified-bg); border-color: var(--verified); }
.field-invalid { background: var(--invalid-bg); border-color: var(--invalid); }
.field-unverifiable { background: var(--unverifiable-bg); border-color: var(--unverifiable); }
.field-human_approved { background: rgba(0,180,216,0.08); border-color: var(--human-approved); }
.field-human_rejected { background: rgba(230,57,70,0.08); border-color: var(--human-rejected); }

.badge {
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    padding: 0.15rem 0.55rem;
    border-radius: 4px;
    white-space: nowrap;
    flex-shrink: 0;
}
.badge-verified { background: var(--verified); color: #000; }
.badge-invalid { background: var(--invalid); color: #fff; }
.badge-unverifiable { background: var(--unverifiable); color: #000; }
.badge-human_approved { background: var(--human-approved); color: #000; }
.badge-human_rejected { background: var(--human-rejected); color: #fff; }

.field-name-label {
    font-size: 0.75rem;
    color: var(--text-dim);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    min-width: 160px;
    flex-shrink: 0;
}
.field-value-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem;
    color: var(--text);
    flex: 1;
}
.field-reason-label {
    font-size: 0.72rem;
    color: var(--text-dim);
    font-style: italic;
    flex: 1.5;
}

/* ── Log entry ── */
.log-entry {
    display: flex;
    gap: 0.75rem;
    padding: 0.4rem 0.75rem;
    border-radius: 6px;
    margin: 0.2rem 0;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    line-height: 1.5;
    border-left: 3px solid transparent;
}
.log-SUCCESS { border-color: var(--verified); background: rgba(0,200,150,0.05); }
.log-WARNING { border-color: var(--unverifiable); background: rgba(255,179,0,0.05); }
.log-ERROR   { border-color: var(--invalid); background: rgba(255,71,87,0.05); }
.log-INFO    { border-color: var(--border); background: rgba(255,255,255,0.02); }
.log-ts { color: var(--text-dim); flex-shrink: 0; width: 165px; }
.log-agent { color: var(--accent); flex-shrink: 0; width: 190px; }
.log-action { color: #a78bfa; flex-shrink: 0; width: 170px; }
.log-detail { color: var(--text); flex: 1; word-break: break-word; }

/* ── Upload box ── */
/* Keep container untouched */
[data-testid="stFileUploader"] {
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
}

/* Style ONLY the dropzone */
[data-testid="stFileUploader"] section {
    background: var(--bg-card2) !important;
    border: 2px dashed var(--border) !important;
    border-radius: 10px !important;
    padding: 1rem !important;
}

/* Fix button overlap explicitly */
[data-testid="stFileUploader"] button {
    position: relative !important;
    z-index: 2 !important;
    width: 100% !important;
}

/* Hide default button text spans */
[data-testid="stFileUploader"] button span {
    font-size: 0 !important;
}

/* Add custom text via pseudo-element */
[data-testid="stFileUploader"] button::after {
    font-size: 0.875rem !important;
}


/* ── Spinner ── */
.stSpinner > div { border-top-color: var(--accent) !important; }

/* ── Human review card ── */
.review-card {
    background: var(--bg-card2);
    border: 1px solid var(--unverifiable);
    border-radius: 10px;
    padding: 1.2rem 1.5rem;
    margin: 0.75rem 0;
}

/* ── Step indicator ── */
.step-indicator {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.5rem 1rem;
    border-radius: 8px;
    background: var(--bg-card2);
    border: 1px solid var(--border);
    font-size: 0.8rem;
    margin: 0.3rem 0;
}
.step-active { border-color: var(--accent); background: rgba(59,130,246,0.1); }
.step-done { border-color: var(--verified); background: rgba(0,200,150,0.07); }

/* ── Input fields ── */
.stTextInput input, .stSelectbox select, .stTextArea textarea {
    background: var(--bg-card2) !important;
    border-color: var(--border) !important;
    color: var(--text) !important;
    font-family: 'Space Grotesk', sans-serif !important;
}

/* scrollbar */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: var(--bg-card); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
</style>
""", unsafe_allow_html=True)


# ─── Session State Defaults ───────────────────────────────────────────────────
def init_session():
    defaults = {
        "uploaded_docs": {},       # {filename: base64_str}
        "verification_result": None,
        "is_verifying": False,
        "selected_doc": None,
        "human_decisions": {},     # {field: {decision, note, reviewer, timestamp}}
        "human_review_done": False,
        "api_key_set": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session()


# ─── Helpers ──────────────────────────────────────────────────────────────────
def img_to_base64(file_bytes: bytes) -> str:
    return base64.b64encode(file_bytes).decode("utf-8")


def status_badge(status: str) -> str:
    labels = {
        "verified": "✓ Verified",
        "invalid": "✗ Invalid",
        "unverifiable": "? Unverifiable",
        "human_approved": "👤 Human Approved",
        "human_rejected": "👤 Human Rejected",
    }
    return labels.get(status, status.upper())


def render_field_row(field_name: str, field_data: dict, human_override: dict = None):
    status = field_data.get("status", "unverifiable")
    if human_override:
        status = "human_approved" if human_override.get("decision") == "approve" else "human_rejected"

    value = field_data.get("value", "—")
    reason = field_data.get("reason", "")
    confidence = field_data.get("confidence", 0)
    display_name = field_name.replace("_", " ").title()

    st.markdown(f"""
    <div class="field-row field-{status}">
        <span class="badge badge-{status}">{status_badge(status)}</span>
        <span class="field-name-label">{display_name}</span>
        <span class="field-value-label">{value or "—"}</span>
        <span class="field-reason-label">{reason}</span>
        <span style="font-size:0.7rem; color: var(--text-dim); flex-shrink:0;">{confidence:.0%}</span>
    </div>
    """, unsafe_allow_html=True)


def render_log_entry(log: dict):
    level = log.get("level", "INFO")
    st.markdown(f"""
    <div class="log-entry log-{level}">
        <span class="log-ts">{log.get('timestamp','')}</span>
        <span class="log-agent">{log.get('agent','')}</span>
        <span class="log-action">{log.get('action','')}</span>
        <span class="log-detail">{log.get('details','')}</span>
    </div>
    """, unsafe_allow_html=True)


def generate_overlay_image(original_b64: str, final_results: dict, human_decisions: dict = None) -> str:
    """Generate image with overlays based on verification status."""
    from PIL import Image, ImageDraw
    import io

    human_decisions = human_decisions or {}

    # Decode original image
    img_data = base64.b64decode(original_b64)
    img = Image.open(io.BytesIO(img_data))
    draw = ImageDraw.Draw(img, 'RGBA')

    width, height = img.size

    # Colors for overlays (with transparency)
    colors = {
        "verified": (0, 200, 150, 100),      # green
        "invalid": (255, 71, 87, 100),       # red
        "unverifiable": (255, 179, 0, 100),  # amber
        "human_approved": (0, 200, 150, 100), # green
        "human_rejected": (255, 71, 87, 100), # red
    }

    for field_name, field_data in final_results.items():
        bbox = field_data.get("bbox")
        if not bbox or len(bbox) != 4:
            continue

        status = field_data.get("status", "unverifiable")
        if field_name in human_decisions:
            decision = human_decisions[field_name].get("decision")
            status = "human_approved" if decision == "approve" else "human_rejected"

        if status not in colors:
            status = "unverifiable"

        # Denormalize bbox
        x1, y1, x2, y2 = bbox
        x1 = int(x1 * width)
        y1 = int(y1 * height)
        x2 = int(x2 * width)
        y2 = int(y2 * height)

        # Draw rectangle
        draw.rectangle([x1, y1, x2, y2], fill=colors[status])

    # Save to bytes
    output = io.BytesIO()
    img.save(output, format='PNG')
    output.seek(0)
    overlay_b64 = base64.b64encode(output.getvalue()).decode('utf-8')
    return overlay_b64


def verdict_html(verdict: str, confidence: float, summary: str) -> str:
    cls_map = {
        "APPROVED": "verdict-approved",
        "REVIEW REQUIRED": "verdict-review",
        "REJECTED": "verdict-rejected",
    }
    icon_map = {"APPROVED": "✅", "REVIEW REQUIRED": "⚠️", "REJECTED": "❌"}
    color_map = {
        "APPROVED": "var(--verified)",
        "REVIEW REQUIRED": "var(--unverifiable)",
        "REJECTED": "var(--invalid)",
    }
    cls = cls_map.get(verdict, "verdict-review")
    icon = icon_map.get(verdict, "⚠️")
    color = color_map.get(verdict, "var(--text)")
    return f"""
    <div class="{cls}">
        <div style="font-size:2rem; margin-bottom:0.3rem">{icon}</div>
        <div style="font-size:1.4rem; font-weight:700; color:{color}; letter-spacing:0.05em">{verdict}</div>
        <div style="font-size:0.85rem; color:var(--text-dim); margin-top:0.4rem">
            Confidence: <b style="color:{color}">{confidence:.0%}</b>
        </div>
        <div style="font-size:0.88rem; color:var(--text); margin-top:0.6rem; line-height:1.6">{summary}</div>
    </div>
    """


# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding: 1rem 0 1.5rem">
        <div style="font-size:1.5rem; font-weight:700; letter-spacing:-0.03em">
            🛡️ <span style="background: linear-gradient(135deg, #3b82f6, #6366f1); -webkit-background-clip:text; -webkit-text-fill-color:transparent">DocVerify</span> AI
        </div>
        <div style="font-size:0.72rem; color:var(--text-dim); letter-spacing:0.1em; text-transform:uppercase; margin-top:0.2rem">
            Autonomous Document Verification
        </div>
    </div>
    """, unsafe_allow_html=True)

    # # Configuration
    # st.markdown('<div class="card-header">Configuration</div>', unsafe_allow_html=True)
    # has_api_key = bool(os.environ.get("GOOGLE_API_KEY"))
    # if has_api_key:
    #     st.success("Google AI API Key loaded from environment.")
    # else:
    #     st.warning(
    #         "Set `GOOGLE_API_KEY` in your `.env` file and restart the app to enable verification."
    #     )

    # st.markdown("---")

    # Upload
    st.markdown('<div class="card-header">Upload Documents</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "",
        type=["png", "jpg", "jpeg", "pdf", "bmp", "tiff"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded:
        for f in uploaded:
            if f.name not in st.session_state.uploaded_docs:
                raw = f.read()
                # Convert PDF first page to image if needed
                if f.name.lower().endswith(".pdf"):
                    try:
                        import fitz  # PyMuPDF
                        doc = fitz.open(stream=raw, filetype="pdf")
                        page = doc.load_page(0)
                        pix = page.get_pixmap(dpi=150)
                        raw = pix.tobytes("jpeg")
                    except ImportError:
                        st.warning("Install PyMuPDF for PDF support: pip install pymupdf")
                        continue
                st.session_state.uploaded_docs[f.name] = img_to_base64(raw)
        st.success(f"✓ {len(st.session_state.uploaded_docs)} document(s) loaded")

    # Document selector
    if st.session_state.uploaded_docs:
        st.markdown('<div class="card-header" style="margin-top:1rem">Select Document</div>', unsafe_allow_html=True)
        doc_names = list(st.session_state.uploaded_docs.keys())
        selected = st.selectbox("Choose document", doc_names, label_visibility="collapsed")
        st.session_state.selected_doc = selected

        if selected:
            b64 = st.session_state.uploaded_docs[selected]
            img_bytes = base64.b64decode(b64)
            img = Image.open(BytesIO(img_bytes))
            st.image(img, caption=selected, use_column_width=True)

    # Clear button
    if st.session_state.uploaded_docs:
        st.markdown("---")
        if st.button("🗑 Clear All Documents", use_container_width=True):
            st.session_state.uploaded_docs = {}
            st.session_state.verification_result = None
            st.session_state.selected_doc = None
            st.session_state.human_decisions = {}
            st.session_state.human_review_done = False
            st.rerun()

    # Agent pipeline info
    st.markdown("---")
    st.markdown('<div class="card-header">Agent Pipeline</div>', unsafe_allow_html=True)
    for agent, icon in [
        ("Orchestrator", "🎯"),
        ("Extraction Agent", "🔍"),
        ("Forgery Detection", "🕵️"),
        ("KYC Agent", "✅"),
        ("Decision Support", "⚖️"),
    ]:
        st.markdown(f"<div style='font-size:0.78rem; color:var(--text-dim); padding:0.2rem 0'>{icon} {agent}</div>",
                    unsafe_allow_html=True)


# ─── Main Content ─────────────────────────────────────────────────────────────
st.markdown("""
<h1 style="font-size:1.8rem; margin-bottom:0.25rem">
    Autonomous Document Verification
</h1>
<p style="color:var(--text-dim); font-size:0.85rem; margin-bottom:1.5rem">
    Multi-agent AI pipeline · Gemini 2.5 · LangGraph Orchestration
</p>
""", unsafe_allow_html=True)


# ─── Verify Button ────────────────────────────────────────────────────────────
col1, col2, col3 = st.columns([2, 1, 2])
with col2:
    can_verify = (
        bool(st.session_state.selected_doc)
        and not st.session_state.is_verifying
    )
    verify_btn = st.button(
        "Verify Document" if not st.session_state.is_verifying else "⏳ Verifying...",
        disabled=not can_verify,
        use_container_width=True,
        type="primary",
    )

if not st.session_state.selected_doc:
    st.info("Upload a document and select it from the sidebar to begin verification.")

# if not has_api_key:
#     st.warning("Set `GOOGLE_API_KEY` in your `.env` file to enable verification.")

# ─── Run Verification ─────────────────────────────────────────────────────────
if verify_btn and can_verify:
    st.session_state.verification_result = None
    st.session_state.human_decisions = {}
    st.session_state.human_review_done = False
    st.session_state.is_verifying = True

    progress_placeholder = st.empty()

    steps = [
        ("🔍 Extracting document fields via OCR...", "ExtractionAgent"),
        ("🕵️ Running forgery & tampering detection...", "ForgeryDetectionAgent"),
        ("✅ Performing KYC compliance checks...", "KYCAgent"),
        ("⚖️ Generating final decision & verdict...", "DecisionSupportAgent"),
    ]

    with progress_placeholder.container():
        st.markdown("### ⏳ Verification In Progress")
        progress_bar = st.progress(0)
        status_text = st.empty()
        log_stream = st.empty()

        for i, (msg, _) in enumerate(steps):
            progress_bar.progress((i) / len(steps))
            status_text.markdown(f"<div class='step-indicator step-active'>🔄 {msg}</div>", unsafe_allow_html=True)

    try:
        from agents.graph import run_verification
        doc_b64 = st.session_state.uploaded_docs[st.session_state.selected_doc]

        result = run_verification(
            document_name=st.session_state.selected_doc,
            document_base64=doc_b64,
        )

        st.session_state.verification_result = result
        st.session_state.is_verifying = False

    except Exception as e:
        st.session_state.is_verifying = False
        st.error(f"❌ Verification failed: {e}")
        st.exception(e)
        progress_placeholder.empty()
        st.stop()

    progress_placeholder.empty()
    st.rerun()


# ─── Results Display ──────────────────────────────────────────────────────────
result = st.session_state.verification_result

if result:
    from utils.logger import logs_to_csv, logs_to_json

    final_results = result.get("final_results", {})
    logs = result.get("logs", [])
    verdict = result.get("overall_verdict", "REVIEW REQUIRED")
    confidence = result.get("overall_confidence", 0.0)
    summary = result.get("overall_summary", "")
    review_fields = result.get("human_review_fields", [])
    human_decisions = st.session_state.human_decisions

    # Determine fields still needing review (red or amber, not yet reviewed)
    fields_needing_review = []
    for fname, fdata in final_results.items():
        status = fdata.get("status", "unverifiable")
        if status in ("invalid", "unverifiable") and fname not in human_decisions:
            fields_needing_review.append(fname)

    tabs = st.tabs(["📊 Verification Results", "🔍 Logs & Audit Trail", "👤 Human Review", "🖼️ Document Overlay"])

    # ─── Tab 1: Results ───────────────────────────────────────────────────────
    with tabs[0]:
        # Verdict banner
        st.markdown(verdict_html(verdict, confidence, summary), unsafe_allow_html=True)

        # Stats row
        verified_count = sum(1 for f in final_results.values() if f.get("status") == "verified")
        invalid_count = sum(1 for f in final_results.values() if f.get("status") == "invalid")
        unverif_count = sum(1 for f in final_results.values() if f.get("status") == "unverifiable")
        total = len(final_results)

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(f"""
            <div class="card" style="text-align:center">
                <div style="font-size:1.8rem; font-weight:700; color:var(--verified)">{verified_count}</div>
                <div class="card-header" style="margin:0">Verified</div>
            </div>""", unsafe_allow_html=True)
        with m2:
            st.markdown(f"""
            <div class="card" style="text-align:center">
                <div style="font-size:1.8rem; font-weight:700; color:var(--invalid)">{invalid_count}</div>
                <div class="card-header" style="margin:0">Invalid</div>
            </div>""", unsafe_allow_html=True)
        with m3:
            st.markdown(f"""
            <div class="card" style="text-align:center">
                <div style="font-size:1.8rem; font-weight:700; color:var(--unverifiable)">{unverif_count}</div>
                <div class="card-header" style="margin:0">Unverifiable</div>
            </div>""", unsafe_allow_html=True)
        with m4:
            reviewed = len(human_decisions)
            st.markdown(f"""
            <div class="card" style="text-align:center">
                <div style="font-size:1.8rem; font-weight:700; color:var(--human-approved)">{reviewed}</div>
                <div class="card-header" style="margin:0">Human Reviewed</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("#### Field-by-Field Verification Results")
        st.markdown("""
        <div style="display:flex; gap:1rem; margin-bottom:0.75rem; flex-wrap:wrap; font-size:0.72rem">
            <span><span class="badge badge-verified">✓ Verified</span> Passed all checks</span>
            <span><span class="badge badge-invalid">✗ Invalid</span> Fraud / KYC failure</span>
            <span><span class="badge badge-unverifiable">? Unverifiable</span> Cannot determine</span>
            <span><span class="badge badge-human_approved">👤 Human Approved</span> Manual override</span>
        </div>
        """, unsafe_allow_html=True)

        # Column headers
        st.markdown("""
        <div style="display:flex; gap:1rem; padding:0.3rem 1rem; font-size:0.68rem;
                    letter-spacing:0.1em; text-transform:uppercase; color:var(--text-dim)">
            <span style="width:120px; flex-shrink:0">Status</span>
            <span style="min-width:160px; flex-shrink:0">Field</span>
            <span style="flex:1">Value</span>
            <span style="flex:1.5">Reason / Finding</span>
            <span style="width:50px; flex-shrink:0; text-align:right">Conf.</span>
        </div>
        """, unsafe_allow_html=True)

        # Sort: verified first, then unverifiable, then invalid
        sort_order = {"verified": 0, "human_approved": 1, "unverifiable": 2, "invalid": 3, "human_rejected": 4}
        sorted_fields = sorted(final_results.items(),
                                key=lambda x: sort_order.get(x[1].get("status", "unverifiable"), 5))

        for fname, fdata in sorted_fields:
            human_override = human_decisions.get(fname)
            render_field_row(fname, fdata, human_override)

        if fields_needing_review:
            st.markdown(f"""
            <div style="margin-top:1rem; padding:0.75rem 1rem; background:rgba(255,179,0,0.08);
                        border:1px solid var(--unverifiable); border-radius:8px; font-size:0.82rem">
                ⚠️ <b>{len(fields_needing_review)} field(s)</b> require human review.
                Go to the <b>👤 Human Review</b> tab to manually verify them.
            </div>
            """, unsafe_allow_html=True)

    # ─── Tab 2: Logs ──────────────────────────────────────────────────────────
    with tabs[1]:
        st.markdown("#### 📋 Verification Audit Trail")

        # Download buttons
        dl1, dl2, _ = st.columns([1, 1, 3])
        with dl1:
            csv_data = logs_to_csv(logs)
            st.download_button(
                "⬇ Download CSV Logs",
                data=csv_data,
                file_name=f"verification_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with dl2:
            json_data = logs_to_json(logs, {k: dict(v) for k, v in final_results.items()}, verdict)
            st.download_button(
                "⬇ Download Full Report",
                data=json_data,
                file_name=f"verification_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True,
            )

        st.markdown("---")

        # Filter
        filter_col1, filter_col2 = st.columns([1, 3])
        with filter_col1:
            log_filter = st.selectbox("Filter by level", ["ALL", "SUCCESS", "WARNING", "ERROR", "INFO"],
                                       label_visibility="collapsed")

        # Stats
        level_counts = {}
        for log in logs:
            l = log.get("level", "INFO")
            level_counts[l] = level_counts.get(l, 0) + 1

        st.markdown(f"""
        <div style="display:flex; gap:1rem; font-size:0.75rem; margin-bottom:0.75rem; flex-wrap:wrap">
            <span style="color:var(--verified)">✓ {level_counts.get('SUCCESS',0)} Success</span>
            <span style="color:var(--unverifiable)">⚠ {level_counts.get('WARNING',0)} Warnings</span>
            <span style="color:var(--invalid)">✗ {level_counts.get('ERROR',0)} Errors</span>
            <span style="color:var(--text-dim)">ℹ {level_counts.get('INFO',0)} Info</span>
            <span style="color:var(--text-dim)">Total: {len(logs)} entries</span>
        </div>
        """, unsafe_allow_html=True)

        # Log header
        st.markdown("""
        <div style="display:flex; gap:0.75rem; padding:0.3rem 0.75rem; font-size:0.65rem;
                    letter-spacing:0.1em; text-transform:uppercase; color:var(--text-dim)">
            <span style="width:165px; flex-shrink:0">Timestamp</span>
            <span style="width:190px; flex-shrink:0">Agent</span>
            <span style="width:170px; flex-shrink:0">Action</span>
            <span>Details</span>
        </div>
        """, unsafe_allow_html=True)

        filtered_logs = logs if log_filter == "ALL" else [l for l in logs if l.get("level") == log_filter]
        for log in filtered_logs:
            render_log_entry(log)

        # Human review decisions in logs
        if human_decisions:
            st.markdown("---")
            st.markdown("#### 👤 Human Review Decisions (Logged)")
            for fname, decision in human_decisions.items():
                log_entry = {
                    "timestamp": decision.get("timestamp", ""),
                    "agent": "HumanReviewer",
                    "action": "HUMAN_DECISION",
                    "details": f"Field '{fname}' → {decision.get('decision','').upper()} by {decision.get('reviewer', 'Reviewer')}. Note: {decision.get('note', 'None')}",
                    "level": "SUCCESS" if decision.get("decision") == "approve" else "WARNING",
                }
                render_log_entry(log_entry)

    # ─── Tab 3: Human Review ──────────────────────────────────────────────────
    with tabs[2]:
        st.markdown("#### 👤 Human-in-the-Loop Review")
        st.markdown("""
        <p style="font-size:0.85rem; color:var(--text-dim)">
        Fields marked as <b style="color:var(--invalid)">Invalid</b> or
        <b style="color:var(--unverifiable)">Unverifiable</b> require manual human review.
        Your decisions will be logged in the audit trail.
        </p>
        """, unsafe_allow_html=True)

        # Get all fields needing review
        all_review_fields = [
            (fname, fdata)
            for fname, fdata in final_results.items()
            if fdata.get("status") in ("invalid", "unverifiable")
        ]

        if not all_review_fields:
            st.success("✅ No fields require human review — all fields were verified automatically.")
        else:
            st.markdown(f"**{len(all_review_fields)} field(s) pending review**")

            reviewer_name = st.text_input(
                "Reviewer Name / ID",
                placeholder="e.g. John Smith / REV-001",
                value=st.session_state.get("reviewer_name", ""),
            )
            st.session_state["reviewer_name"] = reviewer_name

            st.markdown("---")

            for fname, fdata in all_review_fields:
                status = fdata.get("status", "unverifiable")
                value = fdata.get("value", "—")
                reason = fdata.get("reason", "")
                already_reviewed = fname in human_decisions

                status_color = "var(--invalid)" if status == "invalid" else "var(--unverifiable)"
                reviewed_badge = ""
                if already_reviewed:
                    d = human_decisions[fname]
                    rb_color = "var(--verified)" if d["decision"] == "approve" else "var(--invalid)"
                    reviewed_badge = f'<span style="font-size:0.7rem; color:{rb_color}; font-weight:600">{"✓ APPROVED" if d["decision"] == "approve" else "✗ REJECTED"} by {d.get("reviewer","?")}</span>'

                st.markdown(f"""
                <div class="review-card">
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.75rem">
                        <div>
                            <span style="font-size:1rem; font-weight:600">{fname.replace('_',' ').title()}</span>
                            <span style="font-size:0.72rem; color:{status_color}; margin-left:0.5rem; text-transform:uppercase; font-weight:600">{status}</span>
                        </div>
                        {reviewed_badge}
                    </div>
                    <div style="font-family:'JetBrains Mono',monospace; font-size:0.85rem; margin-bottom:0.5rem; color:var(--text)">
                        Value: <b>{value}</b>
                    </div>
                    <div style="font-size:0.78rem; color:var(--text-dim); font-style:italic">
                        Agent finding: {reason}
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Document image alongside for reference
                with st.expander(f"📄 View Document Image for Reference", expanded=False):
                    b64 = st.session_state.uploaded_docs.get(st.session_state.selected_doc, "")
                    if b64:
                        img_bytes = base64.b64decode(b64)
                        img = Image.open(BytesIO(img_bytes))
                        st.image(img, use_column_width=True)
                        # st.image(img, use_container_width=True)

                col_a, col_b, col_c = st.columns([2, 1, 1])
                with col_a:
                    note = st.text_area(
                        f"Review note for {fname}",
                        placeholder="Add your review note or justification...",
                        key=f"note_{fname}",
                        height=68,
                        label_visibility="collapsed",
                    )
                with col_b:
                    approve = st.button(
                        "✅ Approve",
                        key=f"approve_{fname}",
                        use_container_width=True,
                        disabled=not reviewer_name,
                    )
                with col_c:
                    reject = st.button(
                        "❌ Reject",
                        key=f"reject_{fname}",
                        use_container_width=True,
                        disabled=not reviewer_name,
                    )

                if approve or reject:
                    decision = "approve" if approve else "reject"
                    st.session_state.human_decisions[fname] = {
                        "decision": decision,
                        "note": note or "No note provided",
                        "reviewer": reviewer_name or "Unknown",
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "original_status": status,
                        "original_reason": reason,
                    }
                    st.success(f"{'✅ Approved' if decision == 'approve' else '❌ Rejected'}: '{fname}' — logged to audit trail.")
                    st.rerun()

                st.markdown("<hr style='border-color:var(--border); margin:0.5rem 0'>", unsafe_allow_html=True)

            # Summary
            pending = [f for f, _ in all_review_fields if f not in human_decisions]
            if pending:
                st.warning(f"⏳ {len(pending)} field(s) still need review: {', '.join(pending)}")
            else:
                st.success("✅ All flagged fields have been reviewed. Download updated logs from the Audit Trail tab.")

                # Trigger log update for human reviews
                if not st.session_state.human_review_done:
                    st.session_state.human_review_done = True
                    # Add human review entries to logs
                    for fname, dec in st.session_state.human_decisions.items():
                        result["logs"].append({
                            "timestamp": dec["timestamp"],
                            "agent": "HumanReviewer",
                            "action": "HUMAN_DECISION",
                            "details": (
                                f"Field '{fname}' manually {'APPROVED' if dec['decision'] == 'approve' else 'REJECTED'} "
                                f"by '{dec['reviewer']}'. Note: {dec['note']}"
                            ),
                            "level": "SUCCESS" if dec["decision"] == "approve" else "WARNING",
                        })

    # ─── Tab 4: Document Overlay ──────────────────────────────────────────────
    with tabs[3]:
        st.markdown("#### 🖼️ Document with Verification Overlays")
        st.markdown("""
        <p style="font-size:0.85rem; color:var(--text-dim)">
        Verified fields are highlighted in <b style="color:#00c896">green</b>,
        invalid fields in <b style="color:#ff4757">red</b>,
        and unverifiable fields in <b style="color:#ffb300">amber</b>.
        Human-approved fields are also shown in green, human-rejected in red.
        <br><em>Note: Image displayed at original size for accurate overlay positioning.</em>
        </p>
        """, unsafe_allow_html=True)

        # Generate overlay image
        original_b64 = st.session_state.uploaded_docs.get(st.session_state.selected_doc, "")
        if original_b64 and final_results:
            try:
                overlay_b64 = generate_overlay_image(original_b64, final_results, human_decisions)
                st.image(f"data:image/png;base64,{overlay_b64}", caption="Document with Verification Overlays", use_column_width=False)
            except Exception as e:
                st.error(f"Failed to generate overlay: {e}")
        else:
            st.info("No document or verification results available.")

# ─── Empty state ─────────────────────────────────────────────────────────────
elif not st.session_state.is_verifying:
    # Hero info
    st.markdown("""
    <div style="margin-top:2rem">
        <div style="display:grid; grid-template-columns:1fr 1fr 1fr; gap:1rem">
            <div class="card">
                <div style="font-size:1.5rem; margin-bottom:0.5rem">🔍</div>
                <div style="font-weight:600; margin-bottom:0.3rem">OCR Extraction</div>
                <div style="font-size:0.78rem; color:var(--text-dim)">
                    Gemini 2.5 Vision extracts all fields from uploaded documents with high accuracy.
                </div>
            </div>
            <div class="card">
                <div style="font-size:1.5rem; margin-bottom:0.5rem">🕵️</div>
                <div style="font-weight:600; margin-bottom:0.3rem">Forgery Detection</div>
                <div style="font-size:0.78rem; color:var(--text-dim)">
                    Forensic AI checks for pixel manipulation, font inconsistencies, and tampering artifacts.
                </div>
            </div>
            <div class="card">
                <div style="font-size:1.5rem; margin-bottom:0.5rem">✅</div>
                <div style="font-weight:600; margin-bottom:0.3rem">KYC Compliance</div>
                <div style="font-size:0.78rem; color:var(--text-dim)">
                    Validates fields against regulatory rules — formats, expiry, ID patterns, and more.
                </div>
            </div>
        </div>
        <div style="display:grid; grid-template-columns:1fr 1fr; gap:1rem; margin-top:0">
            <div class="card">
                <div style="font-size:1.5rem; margin-bottom:0.5rem">⚖️</div>
                <div style="font-weight:600; margin-bottom:0.3rem">Decision Support</div>
                <div style="font-size:0.78rem; color:var(--text-dim)">
                    Orchestrates all agent results to produce a final verdict: Approved, Review Required, or Rejected.
                </div>
            </div>
            <div class="card">
                <div style="font-size:1.5rem; margin-bottom:0.5rem">👤</div>
                <div style="font-weight:600; margin-bottom:0.3rem">Human-in-the-Loop</div>
                <div style="font-size:0.78rem; color:var(--text-dim)">
                    Flagged fields go to human reviewers. All decisions are logged in the downloadable audit trail.
                </div>
            </div>
        </div>
        <div style="font-size:0.78rem; color:var(--text-dim); text-align:center; margin-top:1rem">
            Powered by <b>Gemini 2.5 Flash</b> · Orchestrated with <b>LangGraph</b> · Built on <b>Streamlit</b>
        </div>
    </div>
    """, unsafe_allow_html=True)
