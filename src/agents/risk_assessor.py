import os
import json
import time
import logging
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from src.ingestion.pdf_parser import extract_text_from_pdf, chunk_text
from src.agents.tools.retriever import get_policy_retriever

load_dotenv()

# Global Logger Setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

#Pydantic Script (strict output schema)
class Violation(BaseModel):
    clause_text: str = Field(description="The exact, verbatim text from the contract clause that triggers the violation.")
    violated_policy: str = Field(description="The exact company rule from the PROVIDED company policies that is being broken. DO NOT invent policies.")
    explanation: str = Field(description="A strict, factual explanation of exactly how the clause contradicts the provided policy.")
    severity: str = Field(description="Categorize the risk strictly as 'Low', 'Medium', or 'High'.")

class RiskReport(BaseModel):
    violations: list[Violation] = Field(default=[], 
        description="A list of policy violations. If the contract complies with the provided policies, or if no policies are relevant, return an empty list.")

# Initialize Groq LLM with Temperature 0.0 for strict, deterministic legal output
logger.info("Initializing Groq LLM (Llama 3.3 70B, Temp 0.0)...")
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv('GROQ_API_KEY'),
    temperature=0.0
)

#Bind pydantic model to llm
structured_llm = llm.with_structured_output(RiskReport)

#template 
template = """You are a strict, objective corporate legal reviewer. Your ONLY job is to compare the Contract Clauses against the provided Company Policies.

STRICT ADJUDICATION RULES:
1. NO ASSUMPTIONS: A violation ONLY exists if the contract text directly and explicitly contradicts a specific constraint (like a numerical limit or a hard ban) in the policy. 
2. NO INFERENCES: Do NOT infer, guess, or assume violations based on what is "implied", "possible", or "not mentioned." For example, if a termination fee is not mentioned, it does NOT violate a 10% fee cap. 
3. NO HALLUCINATIONS: Do not invent legal risk. If the contract complies, or if there is no explicit contradiction, return an empty list.

Company policies:
{company_policies}

Contract Clauses to Analyze:
{contract_clauses}
"""

#Format prompt
prompt = PromptTemplate.from_template(template = template)

#Eexcution Chain
chain = prompt | structured_llm

# Execution Logic
def analyze_contract(file_path: str):
    """Extracts, chunks, retrieves policies, and analyzes a contract for violations."""
    
    # Extract contract text
    logger.info(f"Extracting text from {file_path}...")
    contract_text = extract_text_from_pdf(file_path)
    if not contract_text:
        return "Analysis failed: No text extracted from contract."

    # Chunk the contract text
    logger.info("Chunking contract text...")
    contract_chunks = chunk_text(contract_text)
    logger.info(f"Extracted {len(contract_text)} characters. Split into {len(contract_chunks)} chunks.")

    # Initialize the retriever
    retriever = get_policy_retriever()
    all_violations = []

    for i, chunk in enumerate(contract_chunks):
        logger.info(f"Analyzing chunk {i + 1}/{len(contract_chunks)}...")
        
        # Retrieve
        docs = retriever.invoke(chunk) 
        policy_context = "\n\n".join([doc.page_content for doc in docs])

        if not policy_context.strip():
            logger.info('No policies found for this chunk, skipping.')
            continue

        # Map Inputs
        inputs = {
            'company_policies': policy_context,
            'contract_clauses': chunk
        }
        
        # Execute with Error Handling
        try:
            response = chain.invoke(inputs)
            chunk_data = response.model_dump()
            
            if chunk_data.get("violations"):
                logger.warning(f"Found {len(chunk_data['violations'])} violation(s) in chunk {i + 1}.")
                all_violations.extend(chunk_data["violations"])
            else:
                logger.info("Clear. No violations found.")
                
        except Exception as e:
            logger.error(f"Failed to analyze chunk {i + 1}. Error: {e}")
            
        # Prevent Groq API rate limits
        time.sleep(1) 

    logger.info(f"Analysis complete. Total violations found: {len(all_violations)}")
    
    final_output = {
        "total_violations": len(all_violations), 
        "violations": all_violations
    }
    return json.dumps(final_output, indent=2)

if __name__ == "__main__":
    pdf_path = "data/contracts/sample_contract.pdf" 
    
    logger.info("--- Initiating RAG Risk Analysis ---")
    result = analyze_contract(pdf_path)
    
    print("\n--- Final Aggregated Risk Report ---")
    print(result)

