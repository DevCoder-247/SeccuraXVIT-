import json
import re
from datetime import datetime
from typing import Dict
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from agents.state import VerificationState
from utils.logger import make_log

EXTRACTION_SYSTEM = """You are an expert document information extraction agent.
Your task is to analyze a document image and extract ALL visible text fields and their values, along with their positions.

Extract every identifiable field such as:
- Full Name / Name on Document
- Date of Birth
- Document/ID Number  
- Address
- Issue Date / Date of Issue
- Expiry Date / Valid Until
- Issuing Authority / Issued By
- Document Type (Passport, Aadhaar, PAN, Driving License, Certificate, etc.)
- Nationality / Country
- Gender
- Any other relevant fields visible on the document

For each field, provide:
- value: the extracted text exactly as it appears
- bbox: bounding box as [x1, y1, x2, y2] where coordinates are normalized (0-1) relative to image dimensions

Return ONLY a valid JSON object. No markdown, no backticks, no explanation.
Keys should be clean snake_case field names (e.g. "full_name", "date_of_birth").
If a field is not found or not readable, use null for value and omit bbox.

Example format:
{
  "document_type": {"value": "Aadhaar Card", "bbox": [0.1, 0.2, 0.5, 0.25]},
  "full_name": {"value": "John Doe", "bbox": [0.1, 0.3, 0.6, 0.35]},
  "date_of_birth": {"value": "01/01/1990", "bbox": [0.1, 0.4, 0.4, 0.45]},
  "id_number": {"value": "1234 5678 9012", "bbox": [0.1, 0.5, 0.5, 0.55]},
  "address": {"value": "123 Main St, City", "bbox": [0.1, 0.6, 0.8, 0.7]},
  "gender": {"value": "Male", "bbox": [0.1, 0.75, 0.3, 0.8]},
  "issue_date": null
}"""


def extraction_agent(state: VerificationState) -> dict:
    """Extract fields from document using Gemini Vision."""
    logs = [make_log("ExtractionAgent", "STARTED", "Beginning document field extraction")]

    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0,
        )

        message = HumanMessage(
            content=[
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{state['document_base64']}"
                    },
                },
                {
                    "type": "text",
                    "text": EXTRACTION_SYSTEM,
                },
            ]
        )

        logs.append(make_log("ExtractionAgent", "LLM_CALL", "Sending document to Gemini 2.5 Flash for OCR extraction"))
        response = llm.invoke([message])
        raw = response.content.strip()

        # Strip markdown code fences if present
        raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
        extracted = json.loads(raw)

        # Process extracted fields
        cleaned = {}
        bboxes = {}
        for k, v in extracted.items():
            if v is not None and isinstance(v, dict):
                cleaned[k] = str(v.get("value", ""))
                bboxes[k] = v.get("bbox")
            else:
                cleaned[k] = ""

        doc_type = cleaned.get("document_type", "Unknown Document")
        logs.append(
            make_log(
                "ExtractionAgent",
                "EXTRACTION_COMPLETE",
                f"Extracted {len(cleaned)} fields from '{doc_type}': {', '.join(cleaned.keys())}",
                "SUCCESS",
            )
        )

        return {
            "extracted_fields": cleaned,
            "field_bboxes": bboxes,
            "document_type": doc_type,
            "logs": logs,
            "current_step": "extraction_done",
        }

    except json.JSONDecodeError as e:
        logs.append(make_log("ExtractionAgent", "PARSE_ERROR", f"Failed to parse JSON response: {e}", "ERROR"))
        return {
            "extracted_fields": {},
            "logs": logs,
            "error": f"Extraction JSON parse error: {e}",
            "current_step": "extraction_error",
        }
    except Exception as e:
        logs.append(make_log("ExtractionAgent", "ERROR", str(e), "ERROR"))
        return {
            "extracted_fields": {},
            "logs": logs,
            "error": str(e),
            "current_step": "extraction_error",
        }
