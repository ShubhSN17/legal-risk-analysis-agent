import os
import json
import time
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from src.ingestion.pdf_parser import extract_text_from_pdf, chunk_text

load_dotenv()

#Pydantic Script (strict output schema)
class ClauseRisk(BaseModel):
    clause_reference: str = Field(description="The section number or name...")
    risk_level: str = Field(description="Assess the risk strictly as 'Low','Medium', or 'High'.")
    explanation: str = Field(description="A concise, factual explanation...")

class RiskReport(BaseModel):
    financial_liabilities: list[ClauseRisk] = Field(default=[], description="Risks related to hidden financial liabilities.")
    termination_conditions: list[ClauseRisk] = Field(default=[], description="Risks related to termination conditions.")
    indemnification_requirements: list[ClauseRisk] = Field(default=[], description="Risks related to overly broad indemnification.")

#Initializing LLM
llm = ChatGroq(model="llama-3.3-70b-versatile",api_key = os.getenv('GROQ_API_KEY'))

#Bind pydantic model to llm
structured_llm = llm.with_structured_output(RiskReport)

#template 
template = """You are an expert contract risk analyser. Analyse the risk in the clauses provided below based ONLY on the given data.
Strictly do not give me information from outside of the contract clauses provided.

Contract Clauses to Analyze:
{contract_clauses}
"""

#Format prompt
prompt = PromptTemplate.from_template(template = template)

#Eexcution Chain
chain = prompt | structured_llm

# 3. Execution Logic (Fixed to handle chunks)
def analyze_contract(file_path: str):
    """Extracts, chunks, and analyzes a contract."""
    
    # Extract text using your parser
    full_text = extract_text_from_pdf(file_path)
    if not full_text:
        return "Analysis failed: No text extracted."

    # Chunk the text using your chunker
    chunks = chunk_text(full_text)
    print(f"Extracted {len(full_text)} characters. Split into {len(chunks)} chunks.")
    
    aggregated_report = {
        "financial_liabilities": [],
        "termination_conditions": [],
        "indemnification_requirements": []
    }

    # Iterate through each chunk and run the LLM
    for i, chunk in enumerate(chunks):
        print(f"Analyzing chunk {i + 1}/{len(chunks)}...")
        
        inputs = {"contract_clauses": chunk}
        response = chain.invoke(inputs)
        
        # .model_dump() to get the raw python dictionary from the Pydantic object
        chunk_data = response.model_dump()
        
        # Append the findings from this chunk to our master list
        aggregated_report["financial_liabilities"].extend(chunk_data["financial_liabilities"])
        aggregated_report["termination_conditions"].extend(chunk_data["termination_conditions"])
        aggregated_report["indemnification_requirements"].extend(chunk_data["indemnification_requirements"])

        time.sleep(1)
        
    # Return the final aggregated JSON
    return json.dumps(aggregated_report, indent=2)

if __name__ == "__main__":

    pdf_path = "data/contracts/sample_contract.pdf" 
    
    result = analyze_contract(pdf_path)
    print("\n--- Final Aggregated Risk Report ---")
    print(result)