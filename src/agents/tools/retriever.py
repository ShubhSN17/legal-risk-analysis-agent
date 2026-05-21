import logging
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Global Logger Setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_relevant_context(query: str, db_dir: str = "data/chroma_db") -> str:
    try:
        logger.info("Initializing HuggingFace Embeddings...")
        #Same embedding model used in vector database
        embeddings = HuggingFaceEmbeddings(model_name = "all-MiniLM-L6-v2")

        logger.info("Building connection to the vector database...")
        #connection to stored vector database
        vector_store = Chroma(persist_directory = db_dir, embedding_function = embeddings)

        logger.info("Proccessing Similarity Search...")
        docs = vector_store.similarity_search(query,k=3)

        #loop through docs and extracted page_content
        extracted_text = "\n\n".join([doc.page_content for doc in docs])

        return extracted_text
    except Exception as e:
        logger.error(f"A critical error {e} is occured while searching relevant context..")
        return None



if __name__ == '__main__':
    test_query = "What is the insurance requirement and minimum limits?"
    print(f"Searching for: '{test_query}'\n")
    
    context = get_relevant_context(test_query)
    
    print("--- RETRIEVED CONTEXT ---")
    print(context)