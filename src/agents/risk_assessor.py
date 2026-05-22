import os
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
load_dotenv()

llm = ChatGroq(model="llama-3.3-70b-versatile",api_key = os.getenv('GROQ_API_KEY'))

#template 
template = """You are an expert contract risk analyser. I want you to analyse the risk in the clauses in a contract. 
I will send the related clauses, and you will give me a complete risk report based ONLY on the given data.
Strictly do not give me information from outside of the contract clause provided to you.

The analysis should specifically focus on the following queries:
1. {query1}
2. {query2}
3. {query3}

Contract Clauses to Analyze:
{contract_clauses}
"""

prompt = PromptTemplate.from_template(template = template)

parser = StrOutputParser()

#Eexcution Chain
chain = prompt | llm | parser

queries = {
    "query1": "Are there any hidden financial liabilities?",
    "query2": "What are the termination conditions?",
    "query3": "Are there overly broad indemnification requirements?",
    "contract_clauses": """
        Section 4.1: The Vendor shall indemnify and hold harmless the Client against any and all claims, 
        without limitation, arising from the use of the software. 
        Section 5.2: The Client may terminate this agreement with a 90-day written notice, provided a 
        termination fee of 50% of the remaining contract value is paid.
    """
}

response = chain.invoke(queries)
print(response)