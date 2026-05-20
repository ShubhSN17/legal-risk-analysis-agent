import logging
import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

#Global Logger Setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#Load Environment Variables
load_dotenv()


def store_chunks_in_chroma(chunks: list[str]):
    """
    Embeds text chunks using HuggingFace and stores them persistently in ChromaDB.
    """
    try:
        logger.info("Initializing HuggingFace embeddings...")
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        logger.info("Storing chunks in persistent ChromaDB...")

        #LangChain's Chroma wrapper to handle embeddings and persistence automatically
        vector_store = Chroma.from_texts(texts=chunks,embedding=embeddings,persist_directory="data/chroma_db")

        logger.info(f"Successfully stored {len(chunks)} chunks in data/chroma_db.")

        return vector_store

    except Exception as e:
        logger.exception(f"A critical error occurred while storing chunks: {e}")
        return None
    
#Execution Guard
if __name__ == '__main__':
    #import functions from parser
    from src.ingestion.pdf_parser import extract_text_from_pdf, chunk_text

    test_path = 'data/contracts/sample_contract.pdf'

    print("---- Starting Pipeline test ----")
    raw_text = extract_text_from_pdf(test_path)

    if raw_text:
        print("Text Extracted. Chunking....")
        text_chunks = chunk_text(raw_text)

        print(f"Created {len(text_chunks)} chunks. Sending to vector store....")
        store_chunks_in_chroma(text_chunks)
        print("---Pipeline Complete---")
    else:
        print("Failed to extract Text...")