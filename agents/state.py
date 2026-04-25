from typing import TypedDict, List, Dict, Any, Optional, Annotated
from enum import Enum
import operator


class VerificationStatus(str, Enum):
    VERIFIED = "verified"
    INVALID = "invalid"
    UNVERIFIABLE = "unverifiable"
    PENDING = "pending"
    HUMAN_APPROVED = "human_approved"
    HUMAN_REJECTED = "human_rejected"


class FieldResult(TypedDict):
    value: str
    status: str          # verified / invalid / unverifiable / pending
    reason: str
    agent: str
    confidence: float
    bbox: Optional[List[float]]  # [x1, y1, x2, y2] normalized 0-1


class LogEntry(TypedDict):
    timestamp: str
    agent: str
    action: str
    details: str
    level: str           # INFO / WARNING / ERROR / SUCCESS


class VerificationState(TypedDict):
    # Input
    document_name: str
    document_base64: str
    document_type: str

    # Extraction
    extracted_fields: Dict[str, str]
    field_bboxes: Dict[str, Optional[List[float]]]

    # Agent Results
    forgery_results: Dict[str, FieldResult]
    kyc_results: Dict[str, FieldResult]
    decision_results: Dict[str, FieldResult]

    # Final merged results
    final_results: Dict[str, FieldResult]

    # Overall verdict
    overall_verdict: str       # APPROVED / REVIEW REQUIRED / REJECTED
    overall_confidence: float
    overall_summary: str

    # Human-in-loop
    human_review_fields: List[str]      # fields that need human review
    human_reviews: Dict[str, Dict]      # human decisions per field

    # Logging
    logs: Annotated[List[LogEntry], operator.add]

    # Error handling
    error: Optional[str]
    current_step: str
