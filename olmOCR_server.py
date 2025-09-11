"""
OlmOCR Client - V9 (Final - Robust File Upload & Polling)
"""
import json
import time
from typing import Dict, Any, Optional
from pathlib import Path
import requests
from dataclasses import dataclass, asdict
from datetime import datetime
import os

# --- Data Classes (Unchanged) ---
@dataclass
class ProcessRecord:
    pdf_name: str; is_successful: bool; num_successful_pages: int; num_failed_pages: int
    success_rate: str; elapsed_time: float; start_time: str; finish_time: str
    file_size_mb: float = 0.0; response_size_mb: float = 0.0; error_message: Optional[str] = None

class PersistentReporter:
    def __init__(self, json_filepath: str): self.json_filepath = json_filepath
    def add_record(self, record: ProcessRecord):
        data = []
        if os.path.exists(self.json_filepath):
            try:
                with open(self.json_filepath, 'r', encoding='utf-8') as f: data = json.load(f)
            except (json.JSONDecodeError, FileNotFoundError): pass
        data.append(asdict(record))
        with open(self.json_filepath, 'w', encoding='utf-8') as f: json.dump(data, f, indent=2, ensure_ascii=False)

# --- Main Client Logic ---
class OlmOcrPollingClient:
    def __init__(self, base_url: str, json_report_file: str):
        self.base_url = base_url.rstrip('/')
        self.submit_url = f"{self.base_url}/submit"
        self.status_url = f"{self.base_url}/status"
        self.reporter = PersistentReporter(json_report_file)
        print(f"Polling Client initialized for: {self.base_url}")

    def process_document(self, file_path: str, save_output: bool, output_dir: str, poll_interval_sec: int, total_wait_min: int):
        filename = Path(file_path).name
        start_time_obj = datetime.now()
        print(f"\n{'='*60}\nProcessing: {filename}\n{'='*60}")
        
        try:
            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        except FileNotFoundError:
            print(f"  ✗ File not found: {file_path}")
            return
            
        print(f"  File size: {file_size_mb:.2f} MB")

        # --- Stage 1: Submit the job as a file upload ---
        try:
            print(f"  Submitting job to server as a file upload...")
            with open(file_path, 'rb') as f:
                files = {'file': (filename, f, 'application/pdf')}
                # The timeout for the upload can be longer to accommodate slow connections
                submit_response = requests.post(self.submit_url, files=files, timeout=300) 
            submit_response.raise_for_status()
            task_id = submit_response.json()["task_id"]
            print(f"  ✓ Job submitted successfully. Task ID: {task_id}")
        except requests.RequestException as e:
            print(f"  ✗ Failed to submit job: {e}")
            return

        # --- Stage 2: Poll for the result (Unchanged) ---
        max_polls = (total_wait_min * 60) // poll_interval_sec
        final_result = None
        for i in range(max_polls):
            try:
                print(f"  Polling for result (attempt {i+1}/{max_polls})...", end='\r')
                status_response = requests.get(f"{self.status_url}/{task_id}", timeout=30)
                status_response.raise_for_status()
                data = status_response.json()
                
                if data["status"] == "complete":
                    print(f"\n  ✓ Task complete!                                ")
                    final_result = data["result"]
                    break
                elif data["status"] == "failed":
                    print(f"\n  ✗ Task failed on server: {data['result'].get('error', 'Unknown error')}")
                    final_result = data["result"]
                    break
                time.sleep(poll_interval_sec)
            except requests.RequestException as e:
                print(f"\n  ✗ Polling request failed: {e}")
                time.sleep(poll_interval_sec)
        
        print() # Newline after polling is done
        if final_result is None:
            print(f"  ✗ Job did not complete within the {total_wait_min} minute timeout.")
            final_result = {"error": f"Client-side timeout after {total_wait_min} minutes."}

        # --- Stage 3: Record the outcome (Unchanged) ---
        finish_time_obj = datetime.now()
        elapsed_time = (finish_time_obj - start_time_obj).total_seconds()

        if "error" in final_result:
            record = ProcessRecord(pdf_name=filename, is_successful=False, num_successful_pages=0, num_failed_pages=0,
                success_rate="0.00%", elapsed_time=elapsed_time, start_time=str(start_time_obj), finish_time=str(finish_time_obj),
                file_size_mb=file_size_mb, error_message=final_result.get("error"))
        else:
            completed, failed = final_result.get('completed_pages', 0), final_result.get('failed_pages', 0)
            rate = f"{(completed / (completed + failed) * 100):.2f}%" if (completed + failed) > 0 else "N/A"
            record = ProcessRecord(pdf_name=filename, is_successful=(failed == 0 and completed > 0), num_successful_pages=completed,
                num_failed_pages=failed, success_rate=rate, elapsed_time=elapsed_time, start_time=str(start_time_obj),
                finish_time=str(finish_time_obj), file_size_mb=file_size_mb)
            if save_output and final_result.get('text'):
                output_path = Path(output_dir) / f"{Path(filename).stem}_extracted.md"
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, "w", encoding="utf-8") as f: f.write(final_result["text"])
                print(f"  > Text saved to: {output_path}")
        
        self.reporter.add_record(record)
        print(f"  Total time for {filename}: {elapsed_time:.1f}s. Report updated.")

def main():
    # --- CONFIGURATION ---
    SERVER_URL = "https://8000-01k4t3gxyr35s21y2xjhx28e3e.cloudspaces.litng.ai"
    PDF_INPUT_FOLDER = "H://python_projects//scientific//langextract_pdt_data_extraction//data//"
    OUTPUT_FOLDER = "H://python_projects//scientific//langextract_pdt_data_extraction//data//ocr_results//"
    POLL_INTERVAL_SECONDS = 30
    TOTAL_WAIT_MINUTES = 30
    # --- END CONFIGURATION ---

    Path(PDF_INPUT_FOLDER).mkdir(exist_ok=True)
    Path(OUTPUT_FOLDER).mkdir(exist_ok=True)
    
    client = OlmOcrPollingClient(base_url=SERVER_URL, json_report_file=os.path.join(OUTPUT_FOLDER, "ocr_report.json"))
    
    pdf_files = [f for f in os.listdir(PDF_INPUT_FOLDER) if f.lower().endswith('.pdf')]
    if not pdf_files:
        print(f"No PDF files found in '{PDF_INPUT_FOLDER}'. Please add some PDFs to test.")
        return

    for pdf_filename in pdf_files:
        client.process_document(file_path=os.path.join(PDF_INPUT_FOLDER, pdf_filename), save_output=True,
            output_dir=OUTPUT_FOLDER, poll_interval_sec=POLL_INTERVAL_SECONDS, total_wait_min=TOTAL_WAIT_MINUTES)

    print(f"\nProcessing complete! Check '{os.path.join(OUTPUT_FOLDER, 'ocr_report.json')}' for reports.")

if __name__ == "__main__":
    main()