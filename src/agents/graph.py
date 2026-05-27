import json
import logging
from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, END

#Tools already built
from src.ingestion.pdf_parser import extract_text_from_pdf, chunk_text
from src.agents.risk_assessor import chain, get_policy_retriever

logger = logging.getLogger(__name__)

#State (graph's Memory)
class AgentState(TypedDict):
    file_path: str
    contract_text: str
    chunks: list[str]
    current_index: int
    violations: List[Dict[str, Any]]

#Define Nodes
def ingest_contract(state: AgentState)-> AgentState:
    """Extracts text and chunks the contract."""
    logger.info(f"--- NODE: INGEST CONTRACT ---")
    file_path = state["file_path"]

    text = extract_text_from_pdf(file_path)
    chunks = chunk_text(text)

    logger.info(f"Extracted {len(text)} characters. Split into {len(chunks)} chunks.")

    #Update the state
    return {"contract_text": text, "chunks": chunks, "current_index": 0, "violations": []}

def evaluate_chunk(state: AgentState) -> AgentState:
    """Runs RAG and LLM evaluation on a single chunk."""
    logger.info(f"--- NODE: EVALUATE CHUNK {state['current_index'] + 1}/{len(state['chunks'])} ---")

    current_chunk = state["chunks"][state["current_index"]]
    retriever = get_policy_retriever()

    #RAG Retrieval
    docs = retriever.invoke(current_chunk)
    policy_context = "\n\n".join([doc.page_content for doc in docs])

    current_violations = state.get("violations", [])

    #Efficiency Gate
    if not policy_context.strip():
        logger.info("No policies found for this chunk. Skipping LLM.")
    else:
        #LLM Execution
        inputs = {
            'company_policies': policy_context,
            'contract_clauses': current_chunk
        }
        try:
            response = chain.invoke(inputs)
            chunk_data = response.model_dump()
            if chunk_data.get("violations"):
                logger.warning(f"Found {len(chunk_data['violations'])} violation(s).")
                current_violations.extend(chunk_data["violations"])
            else:
                logger.info("Clear. No violations found.")
        except Exception as e:
            logger.error(f"LLM Error on chunk {state['current_index']}: {e}")

    #Increment index and update state
    return {"violations": current_violations, "current_index": state["current_index"] + 1}

def generate_report(state: AgentState) -> AgentState:
    """Formats the final output."""
    logger.info(f"--- NODE: GENERATE REPORT ---")
    total = len(state["violations"])
    logger.info(f"Analysis complete. Total violations: {total}")
    return state #State remains unchanged, just log the final status

#Define the Edge Logic (The Router)
def route_evaluation(state: AgentState) -> str:
    """Determines whether to loop back or finish."""
    if state["current_index"] < len(state["chunks"]):
        return "evaluate_chunk"
    return "generate_report"

#Build and Compile the Graph
workflow = StateGraph(AgentState)

#Add Nodes
workflow.add_node("ingest_contract", ingest_contract)
workflow.add_node("evaluate_chunk", evaluate_chunk)
workflow.add_node("generate_report", generate_report)

#Add Edges
workflow.set_entry_point("ingest_contract")
workflow.add_edge("ingest_contract", "evaluate_chunk")
workflow.add_conditional_edges(
    "evaluate_chunk",
    route_evaluation,
    {
        "evaluate_chunk": "evaluate_chunk", # Loop back
        "generate_report": "generate_report" # Break loop
    }
)
workflow.add_edge("generate_report", END)

#Compile
app = workflow.compile()

#Execution Guard
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
    
    test_file = "data/contracts/sample_contract.pdf"
    
    # Initialize the graph with the starting state
    initial_state = {
        "file_path": test_file,
        "contract_text": "",
        "chunks": [],
        "current_index": 0,
        "violations": []
    }
    
    print("\n=== KICKING OFF LANGGRAPH AGENT ===\n")
    final_state = app.invoke(initial_state)
    
    print("\n=== FINAL OUTPUT ===")
    report = {
        "total_violations": len(final_state["violations"]),
        "violations": final_state["violations"]
    }
    print(json.dumps(report, indent=2))