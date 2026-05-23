import logging
from pathlib import Path
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

#This ignores low-level DEBUG noise but catches everything else.
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_text_from_pdf(file_path:str)->str:
    

    try:
        target_path = Path(file_path)
        reader = PdfReader(target_path)
        full_text = ''
        for page in reader.pages:
            full_text += page.extract_text() + '\n'

        logger.info(f"Successfully extracted text from {file_path}")

        
    except FileNotFoundError:
        logger.error(f"File not found at path: {file_path}")
        return ""
    except Exception as e:
        logger.error(f'A critical error occured while parsing the pdf: {e}')
        return ""
    
    return full_text

#Chunking
def chunk_text(text: str) -> list[str]:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=6000, chunk_overlap=500)
    chunks = text_splitter.split_text(text)
    return chunks

if __name__ == "__main__":
    # test it locally! 
    test_path = "data/contracts/sample_contract.pdf"
    
    print("Starting extraction test...")
    extracted = extract_text_from_pdf(test_path)
    
    print(f"Total characters extracted: {len(extracted)}")
    chunks = chunk_text(extracted)
    print(len(chunks))
    print(chunks[0])


