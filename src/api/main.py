import os
import shutil
import logging
from fastapi import FastAPI, UploadFile, File, HTTPException

from src.agents.graph import app as agent_app

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

api = FastAPI(title="Legal Risk Agent API")

@api.post("/analyze")
async def analyze_contract(file: UploadFile = File(...)):
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    # We must save the file temporarily because our pdf_parser expects a file path
    temp_file_path = f"data/contracts/temp_{file.filename}"
    
    try:
        #Save uploaded file to disk
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        logger.info(f"Received file for analysis: {file.filename}")

        #Initialize LangGraph State
        initial_state = {
            "file_path": temp_file_path,
            "contract_text": "",
            "chunks": [],
            "current_index": 0,
            "violations": []
        }

        #Execute Graph
        logger.info("Triggering LangGraph execution...")
        final_state = agent_app.invoke(initial_state)

        #Format Response
        report = {
            "total_violations": len(final_state["violations"]),
            "violations": final_state["violations"]
        }
        return report

    except Exception as e:
        logger.error(f"API Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        #ALWAYS clean up the temporary file so the server disk doesn't fill up
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)