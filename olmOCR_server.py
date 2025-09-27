"""
OlmOCR API Server - Improved Version with Auto-Restart Watchdog
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

# --- Global Variables ---
tasks: Dict[str, Dict[str, Any]] = {}
config = OlmOCRConfig()
vllm_process: Optional[subprocess.Popen] = None
vllm_ready = False
last_activity = time.time()  # Track last task completion

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
    global vllm_process, vllm_ready
    
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
    
    vllm_process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env={**os.environ, "OMP_NUM_THREADS": "1"}
    )
    
    asyncio.create_task(monitor_vllm_process())
    await wait_for_vllm_ready()
    vllm_ready = True
    logger.info("vLLM server is ready")

async def monitor_vllm_process():
    if not vllm_process:
        return
    try:
        while vllm_process.poll() is None:
            if vllm_process.stderr:
                line = await asyncio.to_thread(vllm_process.stderr.readline)
                if line:
                    logger.info(f"vLLM: {line.decode().strip()}")
            await asyncio.sleep(1)
    except Exception as e:
        logger.error(f"Error monitoring vLLM process: {e}")

async def wait_for_vllm_ready():
    import httpx
    url = f"http://localhost:{config.vllm_port}/v1/models"
    start_time = time.time()
    max_iterations = config.startup_timeout // config.health_check_interval
    
    for i in range(max_iterations):
        if vllm_process and vllm_process.poll() is not None:
            stdout, stderr = vllm_process.communicate()
            logger.error("vLLM process terminated unexpectedly:")
            logger.error(f"STDOUT: {stdout.decode() if stdout else 'None'}")
            logger.error(f"STDERR: {stderr.decode() if stderr else 'None'}")
            raise RuntimeError("vLLM process terminated unexpectedly")
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(url, timeout=10.0)
                if response.status_code == 200:
                    logger.info(f"vLLM server ready after {time.time() - start_time:.1f}s")
                    return
        except httpx.ConnectError:
            if i % 10 == 0:
                logger.info(f"Still waiting for vLLM server... ({time.time() - start_time:.1f}s elapsed)")
        except Exception as e:
            logger.warning(f"Error checking vLLM status: {e}")
        
        await asyncio.sleep(config.health_check_interval)
    
    raise RuntimeError("vLLM server failed to start in time")

async def stop_vllm_server():
    global vllm_process, vllm_ready
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

# --- Auto-Restart Watchdog ---
async def auto_restart_watchdog():
    global last_activity
    while True:
        await asyncio.sleep(60)  # check every 1 min
        now = time.time()
        active = len([t for t in tasks.values() if t["status"] == "processing"])
        if active == 0 and (now - last_activity) > 1800:  # 30 minutes idle
            logger.info("Idle for 30 min, restarting server...")
            os._exit(0)  # clean exit, wrapper relaunches

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
    global last_activity
    start = time.time()
    while not vllm_ready: 
        await asyncio.sleep(1)
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        try:
            tmp.write(file_content)
            tmp.flush()
            if is_png(tmp.name) or is_jpeg(tmp.name):
                logger.info(f"Converting image {filename} to PDF")
                pdf_bytes = convert_image_to_pdf_bytes(tmp.name)
                tmp.seek(0)
                tmp.truncate()
                tmp.write(pdf_bytes)
                tmp.flush()
            try:
                result = await asyncio.wait_for(
                    process_pdf(tmp.name, filename), 
                    timeout=config.task_timeout
                )
                result["processing_time"] = time.time() - start
                tasks[task_id]["status"] = "complete"
                tasks[task_id]["result"] = result
                last_activity = time.time()  # update activity
                logger.info(f"Task {task_id} completed successfully in {result['processing_time']:.1f}s")
            except asyncio.TimeoutError:
                elapsed = time.time() - start
                error_msg = f"Task timed out after {elapsed:.1f}s"
                tasks[task_id]["status"] = "failed"
                tasks[task_id]["result"] = {"error": error_msg, "timeout": True, "processing_time": elapsed}
        except Exception as e:
            elapsed = time.time() - start
            tasks[task_id]["status"] = "failed"
            tasks[task_id]["result"] = {"error": str(e), "timeout": False, "processing_time": elapsed}
        finally:
            if os.path.exists(tmp.name): 
                os.unlink(tmp.name)

# --- FastAPI App ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting OlmOCR API Server...")
    try:
        await start_vllm_server()
        asyncio.create_task(auto_restart_watchdog())
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
    description="API for OCR processing using OlmOCR model",
    version=VERSION if 'VERSION' in globals() else "1.0.0",
    lifespan=lifespan
)

@app.post("/submit", response_model=SubmitResponse)
async def submit_ocr_task(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    if not vllm_ready:
        raise HTTPException(status_code=503, detail="Server is still starting up")
    task_id = str(uuid.uuid4())
    tasks[task_id] = {"status": "processing", "result": {}}
    content = await file.read()
    background_tasks.add_task(run_ocr_task, task_id, content, file.filename)
    return {"task_id": task_id}

@app.get("/status/{task_id}", response_model=StatusResponse)
async def get_task_status(task_id: str):
    task = tasks.get(task_id)
    if not task: 
        raise HTTPException(status_code=404, detail="Task not found")
    return task

@app.get("/health")
async def health_check():
    status = "healthy" if vllm_ready else "starting"
    return {"status": status, "vllm_ready": vllm_ready, "active_tasks": len([t for t in tasks.values() if t["status"] == "processing"])}

@app.get("/tasks")
async def list_tasks():
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
