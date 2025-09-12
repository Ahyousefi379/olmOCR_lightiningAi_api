"""
OlmOCR Client - Simplified Version
"""
import json
import time
from typing import Dict, Any, Optional, List
from pathlib import Path
import requests
from dataclasses import dataclass, asdict
from datetime import datetime
import os

# --- Data Classes ---
@dataclass
class ProcessRecord:
    pdf_name: str; is_successful: bool; num_successful_pages: int; num_failed_pages: int
    success_rate: str; elapsed_time: float; start_time: str; finish_time: str
    file_size_mb: float = 0.0; response_size_mb: float = 0.0; error_message: Optional[str] = None
    attempt_id: Optional[str] = None

class PersistentReporter:
    def __init__(self, json_filepath: str): 
        self.json_filepath = json_filepath
        
    def add_record(self, record: ProcessRecord):
        data = []
        if os.path.exists(self.json_filepath):
            try:
                with open(self.json_filepath, 'r', encoding='utf-8') as f: 
                    data = json.load(f)
            except (json.JSONDecodeError, FileNotFoundError): 
                pass
        data.append(asdict(record))
        with open(self.json_filepath, 'w', encoding='utf-8') as f: 
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def get_successful_attempts(self, pdf_name: str) -> List[Dict[str, Any]]:
        """Get successful attempts for a specific PDF from the report."""
        if not os.path.exists(self.json_filepath):
            return []
        
        try:
            with open(self.json_filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Filter successful attempts for this PDF
            successful_attempts = [
                record for record in data 
                if record.get('pdf_name') == pdf_name and record.get('is_successful', False)
            ]
            
            return successful_attempts
            
        except (json.JSONDecodeError, FileNotFoundError):
            return []

# --- Main Client Logic ---
class OlmOcrPollingClient:
    def __init__(self, base_url: str, json_report_file: str):
        self.base_url = base_url.rstrip('/')
        self.submit_url = f"{self.base_url}/submit"
        self.status_url = f"{self.base_url}/status"
        self.reporter = PersistentReporter(json_report_file)
        print(f"Polling Client initialized for: {self.base_url}")

    def get_existing_success_count(self, filename: str) -> int:
        """Get the number of existing successful attempts for a file."""
        successful_attempts = self.reporter.get_successful_attempts(filename)
        return len(successful_attempts)

    def process_document(self, file_path: str, save_output: bool, output_dir: str, 
                        poll_interval_sec: int, total_wait_min: int, 
                        attempt_id: Optional[str] = None, force_process: bool = False,
                        output_filename: Optional[str] = None):
        """
        Process a document with OCR.
        
        Args:
            file_path: Path to the PDF file
            save_output: Whether to save the output
            output_dir: Directory to save output
            poll_interval_sec: Polling interval in seconds
            total_wait_min: Total wait time in minutes
            attempt_id: Optional identifier for this attempt
            force_process: If True, skip any internal checks (unused in this version)
            output_filename: Specific filename to save as (optional)
        """
        filename = Path(file_path).name
        start_time_obj = datetime.now()
        
        try:
            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        except FileNotFoundError:
            print(f"  ✗ File not found: {file_path}")
            return False
            
        print(f"  File size: {file_size_mb:.2f} MB")
        if attempt_id:
            print(f"  Attempt ID: {attempt_id}")

        # --- Stage 1: Submit the job as a file upload ---
        try:
            print(f"  Submitting job to server as a file upload...")
            with open(file_path, 'rb') as f:
                files = {'file': (filename, f, 'application/pdf')}
                submit_response = requests.post(self.submit_url, files=files, timeout=300) 
            submit_response.raise_for_status()
            task_id = submit_response.json()["task_id"]
            print(f"  ✓ Job submitted successfully. Task ID: {task_id}")
        except requests.RequestException as e:
            print(f"  ✗ Failed to submit job: {e}")
            self._record_failure(filename, start_time_obj, file_size_mb, f"Submit failed: {e}", attempt_id)
            return False

        # --- Stage 2: Poll for the result ---
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
            error_msg = f"Client-side timeout after {total_wait_min} minutes."
            print(f"  ✗ Job did not complete within the {total_wait_min} minute timeout.")
            final_result = {"error": error_msg}

        # --- Stage 3: Process result ---
        finish_time_obj = datetime.now()
        elapsed_time = (finish_time_obj - start_time_obj).total_seconds()
        
        success = self._process_result(
            final_result, filename, start_time_obj, finish_time_obj, 
            elapsed_time, file_size_mb, save_output, output_dir, 
            attempt_id, output_filename
        )
        
        return success

    def _record_failure(self, filename: str, start_time_obj: datetime, file_size_mb: float, 
                       error_msg: str, attempt_id: Optional[str] = None):
        """Record a failure in the report."""
        finish_time_obj = datetime.now()
        elapsed_time = (finish_time_obj - start_time_obj).total_seconds()
        
        record = ProcessRecord(
            pdf_name=filename, 
            is_successful=False, 
            num_successful_pages=0, 
            num_failed_pages=0,
            success_rate="0.00%", 
            elapsed_time=elapsed_time, 
            start_time=str(start_time_obj), 
            finish_time=str(finish_time_obj),
            file_size_mb=file_size_mb, 
            error_message=error_msg,
            attempt_id=attempt_id
        )
        self.reporter.add_record(record)

    def _process_result(self, final_result: Dict[str, Any], filename: str, start_time_obj: datetime, 
                       finish_time_obj: datetime, elapsed_time: float, file_size_mb: float, 
                       save_output: bool, output_dir: str, attempt_id: Optional[str] = None,
                       output_filename: Optional[str] = None) -> bool:
        """Process the final result and create a record."""
        
        if "error" in final_result:
            record = ProcessRecord(
                pdf_name=filename, 
                is_successful=False, 
                num_successful_pages=0, 
                num_failed_pages=0,
                success_rate="0.00%", 
                elapsed_time=elapsed_time, 
                start_time=str(start_time_obj), 
                finish_time=str(finish_time_obj),
                file_size_mb=file_size_mb, 
                error_message=final_result.get("error"),
                attempt_id=attempt_id
            )
            success = False
        else:
            completed = final_result.get('completed_pages', 0)
            failed = final_result.get('failed_pages', 0)
            rate = f"{(completed / (completed + failed) * 100):.2f}%" if (completed + failed) > 0 else "N/A"
            success = (failed == 0 and completed > 0)
            
            record = ProcessRecord(
                pdf_name=filename, 
                is_successful=success, 
                num_successful_pages=completed,
                num_failed_pages=failed, 
                success_rate=rate, 
                elapsed_time=elapsed_time, 
                start_time=str(start_time_obj),
                finish_time=str(finish_time_obj), 
                file_size_mb=file_size_mb,
                attempt_id=attempt_id
            )
            
            if save_output and final_result.get('text') and success:
                # Determine output filename
                if output_filename:
                    final_output_filename = output_filename
                else:
                    base_name = Path(filename).stem
                    final_output_filename = f"{base_name}_{attempt_id}.md" if attempt_id else f"{base_name}.md"
                
                output_path = Path(output_dir) / final_output_filename
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                with open(output_path, "w", encoding="utf-8") as f: 
                    f.write(final_result["text"])
                print(f"  > Text saved to: {output_path}")
        
        self.reporter.add_record(record)
        
        status_symbol = "✓" if success else "✗"
        print(f"  {status_symbol} Total time for {filename}: {elapsed_time:.1f}s. Report updated.")
        
        return success


def main():
    # --- CONFIGURATION ---
    SERVER_URL = "https://8000-01k4ymt1036tngva09ywdk72z6.cloudspaces.litng.ai"
    PDF_INPUT_FOLDER = "H://python_projects//scientific//langextract_pdt_data_extraction//data//"
    OUTPUT_FOLDER = "H://python_projects//scientific//langextract_pdt_data_extraction//data//ocr_results//"
    POLL_INTERVAL_SECONDS = 30
    TOTAL_WAIT_MINUTES = 30
    # --- END CONFIGURATION ---

    Path(PDF_INPUT_FOLDER).mkdir(exist_ok=True)
    Path(OUTPUT_FOLDER).mkdir(exist_ok=True)
    
    client = OlmOcrPollingClient(
        base_url=SERVER_URL, 
        json_report_file=os.path.join(OUTPUT_FOLDER, "ocr_report.json")
    )
    
    pdf_files = [f for f in os.listdir(PDF_INPUT_FOLDER) if f.lower().endswith('.pdf')]
    if not pdf_files:
        print(f"No PDF files found in '{PDF_INPUT_FOLDER}'. Please add some PDFs to test.")
        return

    for pdf_filename in pdf_files:
        client.process_document(
            file_path=os.path.join(PDF_INPUT_FOLDER, pdf_filename),
            save_output=True,
            output_dir=OUTPUT_FOLDER,
            poll_interval_sec=POLL_INTERVAL_SECONDS,
            total_wait_min=TOTAL_WAIT_MINUTES
        )

    print(f"\nProcessing complete! Check '{os.path.join(OUTPUT_FOLDER, 'ocr_report.json')}' for reports.")

if __name__ == "__main__":
    main()