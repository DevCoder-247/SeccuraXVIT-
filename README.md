# 🛡️ DocVerify AI — Autonomous Document Verification System

A production-grade multi-agent document verification system with a **full backend API layer**.

**Stack:** Streamlit · LangGraph · Gemini 2.5 Flash · FastAPI · Python

---

## Images

### UI
| ![image1](https://github.com/Khilesh-01/SeccuraXVIT-POC/blob/kc-changes/assets/Landing.png) | ![image2](https://github.com/Khilesh-01/SeccuraXVIT-POC/blob/kc-changes/assets/Run.png) |
|-----------------------|-----------------------|

### Results
| ![image1](https://github.com/Khilesh-01/SeccuraXVIT-POC/blob/kc-changes/assets/Verify-review.png) | ![image2](https://github.com/Khilesh-01/SeccuraXVIT-POC/blob/kc-changes/assets/Verify-rejected.png) |
|-----------------------|-----------------------|
| ![image3](https://github.com/Khilesh-01/SeccuraXVIT-POC/blob/kc-changes/assets/Verified-fields.png) | ![image4](https://github.com/Khilesh-01/SeccuraXVIT-POC/blob/kc-changes/assets/log.png) |
|-----------------------|-----------------------|
| ![image5](https://github.com/Khilesh-01/SeccuraXVIT-POC/blob/kc-changes/assets/Human-in-a-loop.png) | ![image6](https://github.com/Khilesh-01/SeccuraXVIT-POC/blob/kc-changes/assets/overlay.png) |
|-----------------------|-----------------------|

---


## 🏗️ Full System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   STREAMLIT FRONTEND                     │
│  Upload → Select → Verify → Results → Human Review      │
└──────────────────────┬──────────────────────────────────┘
                       │
         ┌─────────────▼─────────────┐
         │    LANGGRAPH ORCHESTRATOR  │
         └──────┬────────────────────┘
                │
        ┌───────▼────────┐
        │ Extraction     │  Gemini 2.5 Vision OCR
        │ Agent          │  Extracts all fields from document
        └───────┬────────┘
         ┌──────┴──────┐
         │             │
┌────────▼───┐  ┌──────▼────────────────────────────────┐
│ Forgery    │  │ KYC Agent — TWO PHASES                 │
│ Detection  │  │                                        │
│ Agent      │  │  Phase 1: API Router (Gemini-powered)  │
│            │  │  ┌────────────────────────────────┐    │
│ - Pixels   │  │  │ Decides: which API, which fields│   │
│ - Fonts    │  │  └──────────────┬─────────────────┘   │
│ - Seals    │  │                 │                      │
│ - Layout   │  │  ┌──────────────▼──────────────────┐   │
└────────┬───┘  │  │  FASTAPI BACKEND                 │   │
         │      │  │  /api/college/verify-student     │   │
         │      │  │  /api/government/verify-aadhaar  │   │
         │      │  │  /api/corporate/verify-*         │   │
         │      │  └──────────────┬──────────────────┘   │
         │      │                 │                      │
         │      │  Statuses:      │                      │
         │      │  VALID      → green (verified)         │
         │      │  NOT_FOUND  → red   (invalid)          │
         │      │  UNREACHABLE→ amber (unverifiable)      │
         │      │                 │                      │
         │      │  Phase 2: Rule Engine (remaining fields)│
         │      │  Format/expiry/pattern checks           │
         │      └────────────────────────────────────────┘
         │                        │
         └──────────┬─────────────┘
                    │
          ┌─────────▼────────┐
          │ Decision Support │  Aggregates all results
          │ Agent            │  APPROVED / REVIEW / REJECTED
          └─────────┬────────┘
                    │
          ┌─────────▼──────────────────────────────┐
          │ Human-in-the-Loop Review                │
          │ Red/Amber → manual approve/reject       │
          │ All decisions → append to audit log     │
          └─────────────────────────────────────────┘
```

---

## 📦 Project Structure

```
doc_verifier/
├── app.py                    # Streamlit UI (4 tabs)
├── run.py                    # Launch both services
├── test_api.py               # Integration test script
├── requirements.txt
├── .env.example
│
├── backend/                  # FastAPI Backend
│   ├── main.py               # App + router registration
│   ├── models.py             # Pydantic models
│   ├── data/
│   │   ├── college_db.json   # 5 colleges, 7 students
│   │   └── government_db.json
│   └── routers/
│       ├── college.py        # /api/college/*
│       ├── government.py     # /api/government/*
│       └── corporate.py      # /api/corporate/* (placeholder)
│
├── agents/                   # LangGraph Agents
│   ├── state.py              # Shared state schema (TypedDict)
│   ├── graph.py              # LangGraph DAG & orchestration
│   ├── extraction_agent.py   # Gemini Vision OCR
│   ├── forgery_agent.py      # Forensic analysis
│   ├── kyc_agent.py          # API + Rule-based KYC
│   └── decision_agent.py     # Final verdict
│
└── utils/
    ├── logger.py             # CSV/JSON log export
    ├── api_router.py         # Gemini-powered routing
    └── api_clients/
        ├── base.py           # Abstract BaseAPIClient
        ├── registry.py       # Register all clients here
        ├── college_client.py
        └── government_client.py
```

---

### Step 2 — Create a Virtual Environment (Recommended)

Creating a virtual environment keeps your dependencies isolated.

**Windows (Command Prompt):**

```cmd
python -m venv venv
venv\Scripts\activate
```

**Windows (PowerShell):**

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**macOS / Linux:**

```bash
python3 -m venv venv
source venv/bin/activate
```

> ✅ You should see `(venv)` at the start of your terminal prompt after activation.

---

### Step 3 — Install Dependencies

Install all required packages from `requirements.txt`, plus additional runtime dependencies (`fastapi`, `uvicorn`, `requests`) that are needed but not listed in the file:

```bash
# 1. Install
pip install -r requirements.txt

# 2. Configure
cp .env.example .env       # Add GOOGLE_API_KEY=AIza...

# 3. Run both services
python run.py
```

Once started, you'll see:

```
───────────────────────────────────────────────────
  Services Running
───────────────────────────────────────────────────
  ✓ Backend API  → http://localhost:8000
  ✓ API Docs     → http://localhost:8000/docs
  ✓ Frontend UI  → http://localhost:8501
───────────────────────────────────────────────────
  Press Ctrl+C to stop all services
```

#### Option B — Run Backend Only

```bash
python run.py --backend-only
```

* **API Base URL:** http://localhost:8000
* **Swagger Docs:** http://localhost:8000/docs
* **ReDoc:** http://localhost:8000/redoc

#### Option C — Run Frontend Only

```bash
python run.py --frontend-only
```

* **Streamlit UI:** http://localhost:8501
* ⚠️ API verification calls will show as "unreachable" (amber) without the backend.

#### Custom Ports

```bash
python run.py --port-api 9000 --port-ui 9501
```

---

### Step 6 — Open the Application

1. Open your browser and navigate to: **http://localhost:8501**
2. The Streamlit UI will load with the DocVerify AI interface.
3. If you didn't set the API key in `.env`, add it there and restart the app.

---

### Step 7 — Verify a Document

1. **Upload** a document image (PNG, JPG, JPEG, BMP, TIFF) or PDF via the sidebar.
2. **Select** the document from the dropdown.
3. Click **🚀 Verify Document** in the main area.
4. View results across three tabs:
   - **📊 Verification Results** — Field-by-field status with verdict.
   - **🔍 Logs & Audit Trail** — Full pipeline logs (downloadable as CSV or JSON).
   - **👤 Human Review** — Manually approve/reject flagged fields.

---

### Step 8 — Run Integration Tests (Optional)

Tests verify the backend API endpoints. Start the backend first, then run:

```bash
# Terminal 1: Start the backend
python run.py --backend-only

# Terminal 2: Run tests
python test_api.py
```

Or specify a custom URL:

```bash
python test_api.py --url http://localhost:9000
```

---

## 🧩 Adding a New External API

```python
# Step 1: utils/api_clients/my_client.py
class MyAPIClient(BaseAPIClient):
    @property
    def api_name(self): return "My New API"
    @property
    def document_types(self): return ["driving license", "voter id"]
    def verify(self, extracted_fields: dict) -> APICallResult:
        r = requests.post("https://myapi.com/verify", json=extracted_fields)
        # return APICallResult(status=..., confidence=..., ...)

# Step 2: utils/api_clients/registry.py
REGISTERED_CLIENTS = [
    CollegeAPIClient(),
    MyAPIClient(),   # <- add here, done!
]
```

---

## 🔍 Field Status Logic

| Status         | Color    | Meaning                           |
| -------------- | -------- | --------------------------------- |
| Verified       | 🟢 Green | API found record and fields match |
| Invalid        | 🔴 Red   | NOT_FOUND in DB, or data mismatch |
| Unverifiable   | 🟡 Amber | API unreachable, partial match    |
| Human Approved | 🔵 Blue  | Manual reviewer approved          |

---

## 🏫 Dummy Test Data

**Test students (use these to test verification):**

| Name            | PRN        | College        | Passing Year |
| --------------- | ---------- | -------------- | ------------ |
| Priya Sharma    | 1234567890 | MIT Pune       | 2023         |
| Aditya Kulkarni | 9876543211 | MIT Pune       | 2024         |
| Rohan Mehta     | 5555123456 | VJTI Mumbai    | 2022         |
| Sneha Patil     | 3344556677 | COEP Pune      | 2021         |
| Arjun Nair      | 7788990011 | IIT Bombay     | 2025         |
| Kavya Desai     | 2233445566 | Symbiosis Pune | 2024         |

---

## 🛠️ Troubleshooting

| Issue                                               | Solution                                                                                |
| --------------------------------------------------- | --------------------------------------------------------------------------------------- |
| `ModuleNotFoundError: No module named 'uvicorn'`  | Run `pip install uvicorn`                                                             |
| `ModuleNotFoundError: No module named 'fastapi'`  | Run `pip install fastapi`                                                             |
| `ModuleNotFoundError: No module named 'requests'` | Run `pip install requests`                                                            |
| API key error / Gemini auth failure                 | Verify your `GOOGLE_API_KEY` is valid at [AI Studio](https://aistudio.google.com/apikey) |
| PDF upload shows warning                            | Install PyMuPDF:`pip install pymupdf`                                                 |
| Backend unreachable (amber status)                  | Ensure backend is running (`python run.py` or `--backend-only`)                     |
| Port already in use                                 | Use `--port-api` / `--port-ui` flags to change ports                                |
| `Ctrl+C` doesn't stop services                    | Close the terminal window, or kill Python processes manually                            |

---

## 📄 License

This project is for educational and demonstration purposes.
