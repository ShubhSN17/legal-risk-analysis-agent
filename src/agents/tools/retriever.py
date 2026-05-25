import os
import logging
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)
load_dotenv()

def get_policy_retriever():
    """
    Connects to the persistent Chroma DB and returns a retriever for company policies.
    """
    if not os.path.exists("data/chroma_db"):
        raise RuntimeError("Chroma DB not found. Run vector_store.py first to ingest policies.")

    logger.info("Connecting to ChromaDB for policy retrieval...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    vector_store = Chroma(
        persist_directory="data/chroma_db",
        embedding_function=embeddings,
        collection_name="company_policies"
    )
    
    # Return the top 3 most relevant policy rules
    return vector_store.as_retriever(search_kwargs={"k": 3})

if __name__ == '__main__':
    # Test the retriever
    print("--- Testing Policy Retrieval ---")
    try:
        retriever = get_policy_retriever()
        docs = retriever.invoke("What is our rule on termination fees?")
        print(f"\nRetrieved Rule: {docs[0].page_content}")
        print(f"Source: {docs[0].metadata.get('source')}")
    except Exception as e:
        print(f"Error: {e}")