"""
OlmOCR API Server - V11 (Final - with Empty Output File Check)
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
from fastapi import FastAPI, HTTPException, BackgroundTasks, File, UploadFile
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
    It performs OCR and updates the shared 'tasks' dictionary with the result.
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
        
        logger.info(f"BACKGROUND_TASK [{task_id}]: Executing command: {command}")

        process = subprocess.Popen(
            command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding='utf-8', bufsize=1
        )

        output_lines = []
        if process.stdout:
            for line in iter(process.stdout.readline, ''):
                logger.info(f"OlmOCR Subprocess [{task_id}]: {line.strip()}")
                output_lines.append(line)
            process.stdout.close()

        return_code = process.wait()
        
        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, command, "".join(output_lines))
        
        logger.info(f"BACKGROUND_TASK [{task_id}]: OlmOCR CLI process completed successfully.")

        cli_output = "".join(output_lines)
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
        
        # --- NEW: Check for empty file before parsing ---
        with open(output_files[0], "r", encoding='utf-8') as f:
            first_line = f.readline()
            if not first_line or not first_line.strip():
                raise ValueError("OlmOCR produced an empty or invalid output file, indicating a silent failure on this PDF.")
            
            # If the line is not empty, parse it
            result_data = json.loads(first_line)
        
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

# --- FastAPI App and Endpoints (Unchanged) ---
app = FastAPI()

@app.post("/submit", response_model=SubmitResponse)
async def submit_ocr_task(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    task_id = str(uuid.uuid4())
    tasks[task_id] = {"status": "processing", "result": {}}
    file_content = await file.read()
    background_tasks.add_task(run_ocr_and_update_task, task_id, file_content, file.filename)
    logger.info(f"MAIN_THREAD: Task {task_id} submitted for file {file.filename}.")
    return {"task_id": task_id}

@app.get("/status/{task_id}", response_model=StatusResponse)
async def get_task_status(task_id: str):
    task = tasks.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return task

# --- Main Execution Block (Unchanged) ---
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, timeout_keep_alive=60)