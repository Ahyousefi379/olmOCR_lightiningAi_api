"""
OlmOCR API Server - Improved Version with Auto-Restart Every 30 Minutes
"""

import asyncio
import json
import logging
import os
import shutil
import subprocess
import tempfile
import time
import uuid
from pathlib import Path
from typing import Dict, Any, Optional
from contextlib import asynccontextmanager
from dataclasses import dataclass

import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks, File, UploadFile
from pydantic import BaseModel, Field
from huggingface_hub import snapshot_download
from pypdf import PdfReader

# Import OlmOCR modules
from olmocr.data.renderpdf import render_pdf_to_base64png
from olmocr.prompts.prompts import PageResponse, build_no_anchoring_yaml_prompt
from olmocr.prompts.anchor import get_anchor_text
from olmocr.train.dataloader import FrontMatterParser
from olmocr.image_utils import convert_image_to_pdf_bytes, is_jpeg, is_png
from olmocr.version import VERSION

# --- Configure Logging ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# --- Pydantic Models ---
class SubmitResponse(BaseModel):
    task_id: str = Field(..., description="The unique ID for the processing task.")

class StatusResponse(BaseModel):
    status: str = Field(..., description="The current status of the task (processing, complete, or failed).")
    result: Dict[str, Any] = Field(default_factory=dict, description="The OCR result, only present if status is 'complete'.")

# --- Configuration ---
@dataclass
class OlmOCRConfig:
    model: str = "allenai/olmOCR-7B-0825-FP8"
    vllm_port: int = 30024
    max_page_retries: int = 8
    target_longest_image_dim: int = 1288
    max_page_error_rate: float = 0.004
    gpu_memory_utilization: float = 0.95
    max_model_len: int = 16384
    tensor_parallel_size: int = 1
    startup_timeout: int = 600  # 10 minutes for model download + startup
    health_check_interval: int = 5  # seconds
    task_timeout: int = 240  # 4 minutes per PDF processing task
    restart_interval: int = 1800  # 30 minutes = 1800 seconds

# --- Global Variables ---
tasks: Dict[str, Dict[str, Any]] = {}
config = OlmOCRConfig()
vllm_process: Optional[subprocess.Popen] = None
vllm_ready = False
server_start_time = time.time()
restart_task: Optional[asyncio.Task] = None

# --- GPU Memory Cleanup ---
def cleanup_gpu_memory():
    """Clean up GPU memory to prevent fragmentation issues"""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            logger.info("GPU memory cache cleared")
    except ImportError:
        logger.info("PyTorch not available for GPU cleanup")
    except Exception as e:
        logger.warning(f"GPU cleanup failed: {e}")

# --- VLLM Server Management ---
async def download_model_if_needed():
    """Download model if it doesn't exist locally"""
    model_path = config.model
    if not os.path.exists(model_path) and not model_path.startswith("allenai/"):
        logger.info(f"Model not found locally, attempting to download: {model_path}")
        try:
            model_path = snapshot_download(repo_id=config.model)
            logger.info(f"Model downloaded to: {model_path}")
            return model_path
        except Exception as e:
            logger.error(f"Failed to download model: {e}")
            raise
    return model_path

async def start_vllm_server():
    global vllm_process, vllm_ready, server_start_time, restart_task
    
    # Download model first if needed
    model_path = await download_model_if_needed()

    cmd = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_path,
        "--port", str(config.vllm_port),
        "--disable-log-requests",
        "--served-model-name", "olmocr",
        "--tensor-parallel-size", str(config.tensor_parallel_size),
        "--gpu-memory-utilization", str(config.gpu_memory_utilization),
        "--max-model-len", str(config.max_model_len),
    ]

    logger.info(f"Starting vLLM server with command: {' '.join(cmd)}")
    
    # Start the process
    vllm_process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env={**os.environ, "OMP_NUM_THREADS": "1"}
    )
    
    # Monitor the process output in background
    asyncio.create_task(monitor_vllm_process())

    # Wait for server to be ready
    await wait_for_vllm_ready()
    vllm_ready = True
    server_start_time = time.time()
    logger.info("vLLM server is ready")
    
    # Start the auto-restart timer
    restart_task = asyncio.create_task(schedule_restart())

async def schedule_restart():
    """Schedule automatic restart after the configured interval"""
    try:
        logger.info(f"Auto-restart scheduled in {config.restart_interval} seconds")
        await asyncio.sleep(config.restart_interval)
        
        logger.info("Auto-restart timer triggered")
        await perform_restart()
        
    except asyncio.CancelledError:
        logger.info("Auto-restart timer cancelled")
    except Exception as e:
        logger.error(f"Error in auto-restart scheduler: {e}")

async def perform_restart():
    """Perform the actual restart with proper task handling"""
    global vllm_ready
    
    logger.info("=== INITIATING SERVER RESTART ===")
    
    # Mark server as not ready to prevent new tasks
    vllm_ready = False
    
    # Wait for active tasks to complete (with timeout)
    await wait_for_active_tasks()
    
    # Perform the restart
    await restart_vllm_server()
    
    logger.info("=== SERVER RESTART COMPLETED ===")

async def wait_for_active_tasks(max_wait_time: int = 120):
    """Wait for active processing tasks to complete before restart"""
    start_time = time.time()
    
    while time.time() - start_time < max_wait_time:
        active_tasks = [task for task in tasks.values() if task["status"] == "processing"]
        
        if not active_tasks:
            logger.info("All tasks completed, proceeding with restart")
            return
        
        logger.info(f"Waiting for {len(active_tasks)} active tasks to complete...")
        await asyncio.sleep(10)
    
    active_tasks = [task for task in tasks.values() if task["status"] == "processing"]
    if active_tasks:
        logger.warning(f"Proceeding with restart despite {len(active_tasks)} active tasks (timeout reached)")

async def restart_vllm_server():
    """Stop and restart the vLLM server"""
    global restart_task
    
    # Cancel the current restart timer
    if restart_task and not restart_task.done():
        restart_task.cancel()
    
    # Stop the current server
    await stop_vllm_server()
    
    # Clean up GPU memory
    cleanup_gpu_memory()
    
    # Brief pause to ensure clean shutdown
    await asyncio.sleep(5)
    
    # Start the server again
    await start_vllm_server()

async def monitor_vllm_process():
    """Monitor vLLM process output for debugging"""
    if not vllm_process:
        return
        
    try:
        while vllm_process.poll() is None:
            # Read stderr for error messages
            if vllm_process.stderr:
                line = await asyncio.to_thread(vllm_process.stderr.readline)
                if line:
                    logger.info(f"vLLM: {line.decode().strip()}")
            await asyncio.sleep(1)
    except Exception as e:
        logger.error(f"Error monitoring vLLM process: {e}")

async def wait_for_vllm_ready():
    """Wait for vLLM server to be ready with better error reporting"""
    import httpx
    url = f"http://localhost:{config.vllm_port}/v1/models"
    
    start_time = time.time()
    max_iterations = config.startup_timeout // config.health_check_interval
    
    for i in range(max_iterations):
        # Check if process is still running
        if vllm_process and vllm_process.poll() is not None:
            # Process has terminated
            stdout, stderr = vllm_process.communicate()
            logger.error(f"vLLM process terminated unexpectedly:")
            logger.error(f"STDOUT: {stdout.decode() if stdout else 'None'}")
            logger.error(f"STDERR: {stderr.decode() if stderr else 'None'}")
            raise RuntimeError("vLLM process terminated unexpectedly")
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(url, timeout=10.0)
                if response.status_code == 200:
                    logger.info(f"vLLM server ready after {time.time() - start_time:.1f}s")
                    return
                else:
                    logger.info(f"vLLM server responded with status {response.status_code}")
        except httpx.ConnectError:
            if i % 10 == 0:  # Log every 50 seconds
                elapsed = time.time() - start_time
                logger.info(f"Still waiting for vLLM server... ({elapsed:.1f}s elapsed)")
        except Exception as e:
            logger.warning(f"Error checking vLLM status: {e}")
        
        await asyncio.sleep(config.health_check_interval)
    
    raise RuntimeError(f"vLLM server failed to start within {config.startup_timeout}s timeout")

async def stop_vllm_server():
    global vllm_process, vllm_ready, restart_task
    
    # Cancel restart timer if running
    if restart_task and not restart_task.done():
        restart_task.cancel()
        restart_task = None
    
    if vllm_process:
        logger.info("Stopping vLLM server...")
        vllm_process.terminate()
        try:
            vllm_process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            logger.warning("vLLM server didn't terminate gracefully, killing...")
            vllm_process.kill()
        vllm_process = None
        vllm_ready = False

# --- OCR Processing ---
async def process_page(pdf_path: str, page_num: int) -> Dict[str, Any]:
    import httpx, base64
    from io import BytesIO
    from PIL import Image

    MAX_TOKENS = 4500
    TEMPERATURES = [0.1,0.1,0.2,0.3,0.5,0.8,0.9,1.0]
    attempt, cumulative_rotation = 0, 0

    while attempt < config.max_page_retries:
        try:
            image_base64 = await asyncio.to_thread(
                render_pdf_to_base64png, pdf_path, page_num,
                target_longest_image_dim=config.target_longest_image_dim
            )

            if cumulative_rotation:
                image_bytes = base64.b64decode(image_base64)
                with Image.open(BytesIO(image_bytes)) as img:
                    transpose = {
                        90: Image.Transpose.ROTATE_90,
                        180: Image.Transpose.ROTATE_180,
                        270: Image.Transpose.ROTATE_270
                    }[cumulative_rotation]
                    img = img.transpose(transpose)
                    buf = BytesIO()
                    img.save(buf, format="PNG")
                    image_base64 = base64.b64encode(buf.getvalue()).decode()

            query = {
                "model": "olmocr",
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": build_no_anchoring_yaml_prompt()},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}}
                    ]
                }],
                "max_tokens": MAX_TOKENS,
                "temperature": TEMPERATURES[min(attempt, len(TEMPERATURES)-1)]
            }

            async with httpx.AsyncClient(timeout=60.0) as client:
                r = await client.post(f"http://localhost:{config.vllm_port}/v1/chat/completions", json=query)
            if r.status_code != 200:
                raise ValueError(f"vLLM status {r.status_code}: {r.text}")
            data = r.json()

            model_response = data["choices"][0]["message"]["content"]
            parser = FrontMatterParser(front_matter_class=PageResponse)
            fm, text = parser._extract_front_matter_and_text(model_response)
            pr = parser._parse_front_matter(fm, text)

            if not pr.is_rotation_valid and attempt < config.max_page_retries-1:
                cumulative_rotation = (cumulative_rotation + pr.rotation_correction) % 360
                attempt += 1
                continue

            return {
                "text": pr.natural_text,
                "tokens": {
                    "input": data["usage"].get("prompt_tokens", 0),
                    "output": data["usage"].get("completion_tokens", 0)
                },
                "success": True
            }
        except Exception as e:
            logger.warning(f"Page {page_num} attempt {attempt+1} failed: {e}")
            attempt += 1
            await asyncio.sleep(min(2**attempt, 10))

    try:
        fb_text = get_anchor_text(pdf_path, page_num, pdf_engine="pdftotext")
    except:
        fb_text = ""
    return {"text": fb_text, "tokens": {"input":0,"output":0}, "success": False}

async def process_pdf(pdf_path: str, filename: str) -> Dict[str, Any]:
    reader = PdfReader(pdf_path)
    num_pages = len(reader.pages)

    # Use semaphore to limit concurrent requests to avoid overwhelming the model
    sem = asyncio.Semaphore(4)
    
    async def worker(i): 
        async with sem: 
            return await process_page(pdf_path, i)

    logger.info(f"Processing {num_pages} pages from {filename}")
    results = await asyncio.gather(*[worker(i) for i in range(1, num_pages + 1)])
    
    text = "\n".join(r["text"] for r in results if r.get("text"))
    total_in = sum(r["tokens"]["input"] for r in results)
    total_out = sum(r["tokens"]["output"] for r in results)
    failed = sum(1 for r in results if not r["success"])

    error_rate = failed/num_pages if num_pages else 0
    if error_rate > config.max_page_error_rate:
        raise ValueError(f"Too many failed pages: {failed}/{num_pages}")

    return {
        "text": text,
        "metadata": {"filename": filename, "model_used": config.model},
        "completed_pages": num_pages - failed,
        "failed_pages": failed,
        "page_failure_rate": f"{(failed/num_pages*100):.2f}%" if num_pages else "0.00%",
        "total_input_tokens": total_in,
        "total_output_tokens": total_out,
    }

# --- Background Task ---
async def run_ocr_task(task_id: str, file_content: bytes, filename: str):
    start = time.time()
    
    # Wait for vLLM to be ready
    while not vllm_ready: 
        await asyncio.sleep(1)

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        try:
            tmp.write(file_content)
            tmp.flush()
            
            # Handle image files by converting to PDF
            if is_png(tmp.name) or is_jpeg(tmp.name):
                logger.info(f"Converting image {filename} to PDF")
                pdf_bytes = convert_image_to_pdf_bytes(tmp.name)
                # Overwrite the temp file with PDF content
                tmp.seek(0)
                tmp.truncate()
                tmp.write(pdf_bytes)
                tmp.flush()

            # Process PDF with timeout
            try:
                result = await asyncio.wait_for(
                    process_pdf(tmp.name, filename), 
                    timeout=config.task_timeout
                )
                result["processing_time"] = time.time() - start
                tasks[task_id]["status"] = "complete"
                tasks[task_id]["result"] = result
                logger.info(f"Task {task_id} completed successfully in {result['processing_time']:.1f}s")
                
            except asyncio.TimeoutError:
                elapsed = time.time() - start
                error_msg = f"Task timed out after {elapsed:.1f}s (max: {config.task_timeout}s)"
                logger.error(f"Task {task_id} timed out: {error_msg}")
                tasks[task_id]["status"] = "failed"
                tasks[task_id]["result"] = {
                    "error": error_msg,
                    "timeout": True,
                    "processing_time": elapsed
                }
            
        except Exception as e:
            elapsed = time.time() - start
            logger.error(f"Task {task_id} failed: {str(e)}")
            tasks[task_id]["status"] = "failed"
            tasks[task_id]["result"] = {
                "error": str(e),
                "timeout": False,
                "processing_time": elapsed
            }
        finally:
            if os.path.exists(tmp.name): 
                os.unlink(tmp.name)

# --- FastAPI App ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting OlmOCR API Server with auto-restart...")
    try:
        await start_vllm_server()
        logger.info("OlmOCR API Server startup completed successfully")
    except Exception as e:
        logger.error(f"Failed to start OlmOCR API Server: {e}")
        raise
    
    yield
    
    logger.info("Shutting down OlmOCR API Server...")
    await stop_vllm_server()
    logger.info("OlmOCR API Server shutdown completed")

app = FastAPI(
    title="OlmOCR API Server",
    description="API for OCR processing using OlmOCR model with automatic restart every 30 minutes",
    version=VERSION if 'VERSION' in globals() else "1.0.0",
    lifespan=lifespan
)

@app.post("/submit", response_model=SubmitResponse)
async def submit_ocr_task(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """Submit a PDF or image file for OCR processing"""
    if not vllm_ready:
        raise HTTPException(status_code=503, detail="Server is starting up or restarting, please try again in a moment")
    
    task_id = str(uuid.uuid4())
    tasks[task_id] = {"status": "processing", "result": {}}
    content = await file.read()
    
    logger.info(f"Submitted task {task_id} for file: {file.filename}")
    background_tasks.add_task(run_ocr_task, task_id, content, file.filename)
    return {"task_id": task_id}

@app.get("/status/{task_id}", response_model=StatusResponse)
async def get_task_status(task_id: str):
    """Get the status of a submitted OCR task"""
    task = tasks.get(task_id)
    if not task: 
        raise HTTPException(status_code=404, detail="Task not found")
    return task

@app.get("/health")
async def health_check():
    """Health check endpoint with restart information"""
    uptime = time.time() - server_start_time
    time_to_restart = max(0, config.restart_interval - uptime)
    
    return {
        "status": "healthy" if vllm_ready else "starting/restarting",
        "vllm_ready": vllm_ready,
        "uptime_seconds": round(uptime, 1),
        "next_restart_in_seconds": round(time_to_restart, 1),
        "restart_interval": config.restart_interval,
        "active_tasks": len([t for t in tasks.values() if t["status"] == "processing"]),
        "server_start_time": server_start_time
    }

@app.post("/restart")
async def manual_restart():
    """Manually trigger a server restart"""
    logger.info("Manual restart requested via API")
    
    # Trigger restart in background
    asyncio.create_task(perform_restart())
    
    return {"message": "Server restart initiated, please check /health for status"}

@app.get("/tasks")
async def list_tasks():
    """List all tasks and their statuses"""
    return {
        "total_tasks": len(tasks),
        "processing": len([t for t in tasks.values() if t["status"] == "processing"]),
        "completed": len([t for t in tasks.values() if t["status"] == "complete"]),
        "failed": len([t for t in tasks.values() if t["status"] == "failed"]),
        "tasks": {k: {"status": v["status"]} for k, v in tasks.items()}
    }

# --- Main ---
if __name__ == "__main__":
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000, 
        timeout_keep_alive=1000,
        log_level="info"
    )