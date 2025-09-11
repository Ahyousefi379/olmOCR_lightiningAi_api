"""
OlmOCR API Server - V9 (Final - Robust File Upload & Polling)
"""

import base64
import json
import logging
import os
import re
import shlex
import subprocess
import tempfile
import time
import uuid
from pathlib import Path
from typing import Dict, Any

import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks, File, UploadFile, Form
from pydantic import BaseModel, Field

# --- Configure Logging ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# --- Pydantic Models ---
class SubmitResponse(BaseModel):
    task_id: str = Field(..., description="The unique ID for the processing task.")

class StatusResponse(BaseModel):
    status: str = Field(..., description="The current status of the task (processing, complete, or failed).")
    result: Dict[str, Any] = Field(default_factory=dict, description="The OCR result, only present if status is 'complete'.")

# --- In-Memory Task Storage ---
tasks: Dict[str, Dict[str, Any]] = {}

# --- OCR Processing Function ---
def run_ocr_and_update_task(task_id: str, file_content: bytes, filename: str):
    """
    This is the core blocking function that runs in the background.
    It performs OCR on the provided file content.
    """
    start_time = time.time()
    request_id = str(uuid.uuid4())
    TEMP_DIR = Path(tempfile.gettempdir()) / "olmocr_processing"
    input_dir = TEMP_DIR / "inputs" / request_id
    workspace_dir = TEMP_DIR / "workspaces" / request_id
    
    try:
        input_dir.mkdir(parents=True, exist_ok=True)
        workspace_dir.mkdir(parents=True, exist_ok=True)
        input_file_path = input_dir / filename

        with open(input_file_path, "wb") as f: f.write(file_content)
        
        latest_model = "allenai/olmOCR-7B-0725-FP8"
        command = (
            f"python -m olmocr.pipeline {shlex.quote(str(workspace_dir))} "
            f"--pdfs {shlex.quote(str(input_file_path))} "
            f"--model {shlex.quote(latest_model)}"
        )
        
        process_timeout = 7000
        logger.info(f"BACKGROUND_TASK [{task_id}]: Executing command...")
        process = subprocess.run(command, shell=True, capture_output=True, text=True, check=True, timeout=process_timeout)
        logger.info(f"BACKGROUND_TASK [{task_id}]: OlmOCR CLI process completed.")

        cli_output = process.stdout + "\n" + process.stderr
        stats = {"completed_pages": 0, "failed_pages": 0, "page_failure_rate": "0.00%"}
        completed_match = re.search(r"Completed pages: (\d+)", cli_output)
        if completed_match: stats["completed_pages"] = int(completed_match.group(1))
        failed_match = re.search(r"Failed pages: (\d+)", cli_output)
        if failed_match: stats["failed_pages"] = int(failed_match.group(1))
        rate_match = re.search(r"Page Failure rate: ([\d.]+%)", cli_output)
        if rate_match: stats["page_failure_rate"] = rate_match.group(1)
        
        logger.info(f"BACKGROUND_TASK [{task_id}]: Parsed stats: {stats}")

        results_dir = workspace_dir / "results"
        output_files = list(results_dir.glob("output_*.jsonl"))
        if not output_files: raise FileNotFoundError("OlmOCR did not produce an output file.")
        
        with open(output_files[0], "r", encoding='utf-8') as f: result_data = json.loads(f.readline())
        
        tasks[task_id]['status'] = 'complete'
        tasks[task_id]['result'] = {
            "text": result_data.get("text", ""),
            "metadata": {"filename": filename, "model_used": latest_model},
            "processing_time": time.time() - start_time, **stats,
        }

    except Exception as e:
        logger.error(f"BACKGROUND_TASK [{task_id}]: Task failed: {e}")
        tasks[task_id]['status'] = 'failed'
        tasks[task_id]['result'] = {'error': str(e)}
    finally:
        import shutil
        if input_dir.exists(): shutil.rmtree(input_dir)
        if workspace_dir.exists(): shutil.rmtree(workspace_dir)
        logger.info(f"BACKGROUND_TASK [{task_id}]: Cleaned up directories.")

# --- FastAPI App and Endpoints ---
app = FastAPI()

@app.post("/submit", response_model=SubmitResponse)
async def submit_ocr_task(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """
    Accepts a file upload (`multipart/form-data`), starts the OCR job in the 
    background, and immediately returns a task ID.
    """
    task_id = str(uuid.uuid4())
    tasks[task_id] = {"status": "processing", "result": {}}
    
    # Read the file content into memory. FastAPI handles large files efficiently.
    file_content = await file.read()
    
    background_tasks.add_task(run_ocr_and_update_task, task_id, file_content, file.filename)
    
    logger.info(f"MAIN_THREAD: Task {task_id} submitted for file {file.filename}.")
    return {"task_id": task_id}

@app.get("/status/{task_id}", response_model=StatusResponse)
async def get_task_status(task_id: str):
    """Allows the client to poll for the status and result of a task."""
    task = tasks.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    return task

# --- Main Execution Block ---
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, timeout_keep_alive=60)