from langgraph.graph import StateGraph, END
from agents.state import VerificationState
from agents.extraction_agent import extraction_agent
from agents.forgery_agent import forgery_detection_agent
from agents.kyc_agent import kyc_agent
from agents.decision_agent import decision_support_agent
from utils.logger import make_log
import asyncio
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor


def should_continue_after_extraction(state: VerificationState) -> str:
    """Route after extraction — skip rest if error."""
    if state.get("error") or not state.get("extracted_fields"):
        return "end"
    return "continue"


def parallel_verification_agents(state: VerificationState) -> dict:
    """
    Run forgery detection and KYC verification in parallel.
    Both agents are independent and only depend on extracted_fields.
    """
    # Run both agents concurrently using ThreadPoolExecutor
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        # Submit both agents to run in parallel
        forgery_future = executor.submit(forgery_detection_agent, state)
        kyc_future = executor.submit(kyc_agent, state)
        
        # Wait for both to complete
        forgery_result = forgery_future.result()
        kyc_result = kyc_future.result()
    
    # Merge results
    merged_logs = state.get("logs", [])
    merged_logs.extend(forgery_result.get("logs", []))
    merged_logs.extend(kyc_result.get("logs", []))
    
    return {
        "forgery_results": forgery_result.get("forgery_results", {}),
        "kyc_results": kyc_result.get("kyc_results", {}),
        "logs": merged_logs,
    }


def build_verification_graph():
    """Build and compile the LangGraph verification workflow with parallel agents."""
    workflow = StateGraph(VerificationState)

    # Add nodes
    workflow.add_node("extraction", extraction_agent)
    workflow.add_node("parallel_agents", parallel_verification_agents)  # Combines forgery + kyc in parallel
    workflow.add_node("decision_support", decision_support_agent)

    # Entry point
    workflow.set_entry_point("extraction")

    # Conditional routing after extraction
    workflow.add_conditional_edges(
        "extraction",
        should_continue_after_extraction,
        {
            "continue": "parallel_agents",
            "end": END,
        },
    )

    # Run parallel agents, then decision support
    workflow.add_edge("parallel_agents", "decision_support")
    workflow.add_edge("decision_support", END)

    return workflow.compile()


# Singleton graph instance
_graph = None


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_verification_graph()
    return _graph


def run_verification(document_name: str, document_base64: str) -> VerificationState:
    """Run the full verification pipeline on a document."""
    graph = get_graph()

    initial_state: VerificationState = {
        "document_name": document_name,
        "document_base64": document_base64,
        "document_type": "Unknown",
        "extracted_fields": {},
        "forgery_results": {},
        "kyc_results": {},
        "decision_results": {},
        "final_results": {},
        "overall_verdict": "PENDING",
        "overall_confidence": 0.0,
        "overall_summary": "",
        "human_review_fields": [],
        "human_reviews": {},
        "logs": [make_log("Orchestrator", "WORKFLOW_START",
                          f"Starting verification pipeline for document: {document_name}")],
        "error": None,
        "current_step": "init",
    }

    result = graph.invoke(initial_state)
    result["logs"].append(
        make_log("Orchestrator", "WORKFLOW_COMPLETE",
                 f"Verification pipeline complete. Verdict: {result.get('overall_verdict', 'N/A')}",
                 "SUCCESS" if result.get("overall_verdict") == "APPROVED" else "WARNING")
    )
    return result
