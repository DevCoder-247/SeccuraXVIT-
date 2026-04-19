import json
import re
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from agents.state import VerificationState, FieldResult
from utils.logger import make_log

DECISION_SYSTEM = """You are the Decision Support Agent for a document verification system.
You receive results from two specialist agents — a Forgery Detection Agent and a KYC Agent.
Your task is to produce a final field-level verdict and an overall document decision.

Decision Logic:
1. If BOTH agents mark a field as valid/verified → Final: "verified"
2. If EITHER agent marks a field as "invalid" (fraud flagged OR KYC failed) → Final: "invalid"
3. If either agent marks "unverifiable" but no "invalid" → Final: "unverifiable"
4. If a field was only checked by one agent → Use that agent's result

Overall Document Verdict:
- "APPROVED": All critical fields verified, no forgery detected
- "REVIEW REQUIRED": Some fields unverifiable or minor issues detected (needs human review)
- "REJECTED": Critical fields invalid, forgery detected, or document expired

Critical fields (if present): full_name, id_number, date_of_birth, expiry_date, issuing_authority

Return ONLY valid JSON. No markdown, no explanation.
Format:
{
  "field_decisions": {
    "field_name": {
      "final_status": "verified" | "invalid" | "unverifiable",
      "confidence": 0.0-1.0,
      "reasoning": "Combined analysis from both agents",
      "forgery_input": "summary of forgery agent finding",
      "kyc_input": "summary of KYC agent finding"
    }
  },
  "overall_verdict": "APPROVED" | "REVIEW REQUIRED" | "REJECTED",
  "overall_confidence": 0.0-1.0,
  "overall_summary": "2-3 sentence executive summary of verification outcome",
  "critical_issues": ["list", "of", "critical", "problems"],
  "fields_needing_human_review": ["fields", "that", "need", "manual", "review"]
}"""


def decision_support_agent(state: VerificationState) -> dict:
    """Aggregate results and produce final verification decision."""
    logs = [make_log("DecisionSupportAgent", "STARTED", "Aggregating agent results for final decision")]

    if state.get("error"):
        logs.append(make_log("DecisionSupportAgent", "SKIPPED", "Skipping due to prior error", "WARNING"))
        return {"decision_results": {}, "final_results": {}, "logs": logs}

    forgery = state.get("forgery_results", {})
    kyc = state.get("kyc_results", {})
    extracted = state.get("extracted_fields", {})

    # Build combined summary for LLM
    all_fields = set(list(forgery.keys()) + list(kyc.keys()))
    # Remove meta-fields from forgery agent
    all_fields.discard("overall_document_integrity")

    field_summary = {}
    for field in all_fields:
        entry = {"field_value": extracted.get(field, "")}
        if field in forgery:
            f = forgery[field]
            entry["forgery"] = {"status": f.get("status"), "reason": f.get("reason"), "confidence": f.get("confidence")}
        if field in kyc:
            k = kyc[field]
            entry["kyc"] = {"status": k.get("status"), "reason": k.get("reason"), "confidence": k.get("confidence")}
        field_summary[field] = entry

    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0,
        )

        prompt = f"""{DECISION_SYSTEM}

Document Type: {state.get('document_type', 'Unknown')}

Combined agent results:
{json.dumps(field_summary, indent=2)}

Produce the final verification decision."""

        logs.append(make_log("DecisionSupportAgent", "LLM_CALL",
                              f"Running decision logic across {len(all_fields)} fields"))
        message = HumanMessage(content=prompt)
        response = llm.invoke([message])
        raw = response.content.strip()
        raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
        decision = json.loads(raw)

        field_decisions = decision.get("field_decisions", {})
        final_results: dict[str, FieldResult] = {}
<<<<<<< HEAD
=======
        bboxes = state.get("field_bboxes", {})
>>>>>>> 9172dd2 (Frontend Updates)

        for field, data in field_decisions.items():
            status = data.get("final_status", "unverifiable")
            confidence = float(data.get("confidence", 0.5))
            reasoning = data.get("reasoning", "")

            final_results[field] = FieldResult(
                value=extracted.get(field, ""),
                status=status,
                reason=reasoning,
                agent="DecisionSupportAgent",
                confidence=confidence,
<<<<<<< HEAD
=======
                bbox=bboxes.get(field),
>>>>>>> 9172dd2 (Frontend Updates)
            )

            icon = {"verified": "✓", "invalid": "✗", "unverifiable": "?"}.get(status, "?")
            level = {"verified": "SUCCESS", "invalid": "WARNING", "unverifiable": "WARNING"}.get(status, "INFO")
            logs.append(make_log("DecisionSupportAgent", f"FIELD_{status.upper()}",
                                  f"{icon} '{field}' → {status.upper()} (confidence: {confidence:.0%}): {reasoning}", level))

        verdict = decision.get("overall_verdict", "REVIEW REQUIRED")
        confidence = float(decision.get("overall_confidence", 0.5))
        summary = decision.get("overall_summary", "")
        issues = decision.get("critical_issues", [])
        review_fields = decision.get("fields_needing_human_review", [])

        if issues:
            logs.append(make_log("DecisionSupportAgent", "CRITICAL_ISSUES",
                                  f"Critical issues: {'; '.join(issues)}", "WARNING"))

        verdict_level = {"APPROVED": "SUCCESS", "REVIEW REQUIRED": "WARNING", "REJECTED": "ERROR"}.get(verdict, "INFO")
        logs.append(make_log("DecisionSupportAgent", "FINAL_VERDICT",
                              f"Overall verdict: {verdict} (confidence: {confidence:.0%}) — {summary}", verdict_level))

        return {
            "decision_results": field_decisions,
            "final_results": final_results,
            "overall_verdict": verdict,
            "overall_confidence": confidence,
            "overall_summary": summary,
            "human_review_fields": review_fields,
            "logs": logs,
            "current_step": "decision_done",
        }

    except json.JSONDecodeError as e:
        logs.append(make_log("DecisionSupportAgent", "PARSE_ERROR", f"Failed to parse decision: {e}", "ERROR"))
        return {
            "decision_results": {},
            "final_results": {},
            "overall_verdict": "REVIEW REQUIRED",
            "overall_confidence": 0.0,
            "overall_summary": "Decision agent encountered an error.",
            "human_review_fields": [],
            "logs": logs,
            "error": str(e),
        }
    except Exception as e:
        logs.append(make_log("DecisionSupportAgent", "ERROR", str(e), "ERROR"))
        return {
            "decision_results": {},
            "final_results": {},
            "overall_verdict": "REVIEW REQUIRED",
            "overall_confidence": 0.0,
            "overall_summary": "System error during decision phase.",
            "human_review_fields": [],
            "logs": logs,
            "error": str(e),
        }
