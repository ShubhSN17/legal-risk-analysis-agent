import logging
import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

#Global Logger Setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#Load Environment Variables
load_dotenv()

#HuggingFace embeddings (local)
def get_embedding_model():
    """Initializes and returns the local HuggingFace embedding model."""
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def store_policies_in_chroma(chunks: list[str], source_name: str):
    """
    Embeds POLICY text chunks and stores them persistently.
    """
    try:
        logger.info("Initializing HuggingFace embeddings...")
        embeddings = get_embedding_model()

        # Add metadata so the LLM knows exactly which policy document this rule came from
        metadatas = [{"source": source_name} for _ in chunks]

        logger.info("Storing policy chunks in persistent ChromaDB...")

        vector_store = Chroma.from_texts(
            texts=chunks,
            embedding=embeddings,
            metadatas=metadatas,
            persist_directory="data/chroma_db",
            collection_name="company_policies"
        )

        logger.info(f"Successfully stored {len(chunks)} chunks in data/chroma_db.")
        return vector_store

    except Exception as e:
        logger.exception(f"A critical error occurred while storing chunks: {e}")
        return None
    
# Execution Guard
if __name__ == '__main__':
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    
    # We are testing POLICY ingestion, NOT contract ingestion
    test_policy_path = 'data/policies/company_playbook.txt'
    
    print("---- Starting Policy Ingestion Test ----")
    
    if os.path.exists(test_policy_path):
        with open(test_policy_path, 'r') as file:
            raw_policy_text = file.read()
            
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        policy_chunks = text_splitter.split_text(raw_policy_text)

        print(f"Created {len(policy_chunks)} policy chunks. Sending to vector store....")
        store_policies_in_chroma(policy_chunks, source_name="company_playbook.txt")
        print("--- Ingestion Complete ---")