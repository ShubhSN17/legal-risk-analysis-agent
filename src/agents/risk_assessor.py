import os
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from dotenv import load_dotenv
load_dotenv()

#Pydantic Script (strict output schema)
class ClauseRisk(BaseModel):
    clause_reference: str = Field(description="The section number or name...")
    risk_level: str = Field(description="Assess the risk strictly as 'Low','Medium', or 'High'.")
    explanation: str = Field(description="A concise, factual explanation...")

class RiskReport(BaseModel):
    financial_liabilities: list[ClauseRisk] = Field(description="Risks related to hidden financial liabilities.")
    termination_conditions: list[ClauseRisk] = Field(description="Risks related to termination conditions.")
    indemnification_requirements: list[ClauseRisk] = Field(description="Risks related to overly broad indemnification.")

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

#Input Data
inputs = {
    "contract_clauses": """
        Section 4.1: The Vendor shall indemnify and hold harmless the Client against any and all claims, 
        without limitation, arising from the use of the software. 
        Section 5.2: The Client may terminate this agreement with a 90-day written notice, provided a 
        termination fee of 50% of the remaining contract value is paid.
    """
}

#Invoking chain
response = chain.invoke(inputs)

#using model_dump_json() to cleanly print the Pydantic object as a JSON string
print(response.model_dump_json(indent=2))