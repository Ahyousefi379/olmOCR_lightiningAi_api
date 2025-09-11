"""
OlmOCR Client Helper (Updated for V3 Stats Server) - WITH PERSISTENT JSON REPORTING

This client is designed to interact with the OlmOCR API server that acts as
a wrapper around the official command-line tool and returns processing statistics.

Features:
- Simple progress printing
- Persistent JSON reporting (append mode)
- Multiple runs tracking per PDF
- Comprehensive process tracking
"""
import base64
import json
import time
from typing import Dict, Any, Optional
from pathlib import Path
import requests
from dataclasses import dataclass, asdict
from datetime import datetime
import os

@dataclass
class ProcessRecord:
    """Record for a single processing attempt"""
    pdf_name: str
    is_successful: bool  # True only if all pages processed successfully (failed_pages == 0)
    num_successful_pages: int
    num_failed_pages: int
    success_rate: str
    elapsed_time: float
    start_time: str
    finish_time: str
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)

class PersistentReporter:
    """Handles persistent JSON reporting for OCR operations"""
    
    def __init__(self, json_filepath: str):
        self.json_filepath = json_filepath
        
    def add_record(self, record: ProcessRecord):
        """Add a processing record to the JSON file"""
        # Read existing data
        if os.path.exists(self.json_filepath):
            try:
                with open(self.json_filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                data = []
        else:
            data = []
        
        # Append new record
        data.append(record.to_dict())
        
        # Write back to file
        with open(self.json_filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

class OlmOcrClient:
    """
    Client for the OlmOCR Wrapper API (V3 with Stats) - Enhanced with persistent JSON reporting.
    """
    def __init__(self, base_url: str = "https://8000-01k4t3gxyr35s21y2xjhx28e3e.cloudspaces.litng.ai", 
                 json_report_file: str = "olmocr_report.json"):
        """
        Initialize the client with the API base URL and report file.
        
        Args:
            base_url (str): The API server URL
            json_report_file (str): Path to the JSON report file (will be created if doesn't exist)
        """
        self.base_url = base_url.rstrip('/')
        self.predict_url = f"{self.base_url}/predict"
        self.reporter = PersistentReporter(json_report_file)
        self.json_report_file = json_report_file
        print(f"Client initialized for server at: {self.base_url}")
        print(f"Reports will be saved to: {json_report_file}")
        
        # Test server connection
        self._test_connection()

    def _test_connection(self):
        """Test if the server is reachable"""
        try:
            # Try a simple GET request to the base URL
            response = requests.get(self.base_url, timeout=10)
            print(f"Server connection test: Status {response.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"Warning: Server connection test failed: {e}")
            print("This may indicate the server is not running or not accessible.")

    def _encode_file_to_base64(self, file_path: str) -> str:
        """
        Encode a file to a base64 string.
        """
        try:
            with open(file_path, "rb") as file:
                return base64.b64encode(file.read()).decode('utf-8')
        except FileNotFoundError:
            print(f"Error: File not found at {file_path}")
            return None

    def process_document(self, file_path: str, save_output: bool = False, output_path: str = None) -> Dict[str, Any]:
        """
        Processes a PDF or image file by sending it to the wrapper server.
        Enhanced with simplified printing and persistent JSON reporting.

        Args:
            file_path (str): Path to the PDF or image file.
            save_output (bool): Whether to save extracted text to file
            output_path (str): Path to save output (optional)

        Returns:
            Dict[str, Any]: API response with extracted text and metadata.
        """
        filename = Path(file_path).name
        start_time = datetime.now()
        
        # Simple progress print
        print(f"{filename} is processing...")

        file_content = self._encode_file_to_base64(file_path)
        if not file_content:
            # Create error record
            finish_time = datetime.now()
            elapsed_time = (finish_time - start_time).total_seconds()
            
            error_record = ProcessRecord(
                pdf_name=filename,
                is_successful=False,
                num_successful_pages=0,
                num_failed_pages=0,
                success_rate="0.00%",
                elapsed_time=elapsed_time,
                start_time=start_time.strftime("%Y-%m-%d %H:%M:%S"),
                finish_time=finish_time.strftime("%Y-%m-%d %H:%M:%S"),
                error_message="File encoding failed - file not found or unreadable"
            )
            self.reporter.add_record(error_record)
            return {"error": "File encoding failed", "file": file_path}

        payload = {
            "content": file_content,
            "filename": filename,
        }
        
        processing_start = time.time()
        
        try:
            response = requests.post(self.predict_url, json=payload, timeout=2000)
            finish_time = datetime.now()
            elapsed_time = (finish_time - start_time).total_seconds()

            if response.status_code == 200:
                result = response.json()
                completed_pages = result.get('completed_pages', 0)
                failed_pages = result.get('failed_pages', 0)
                page_failure_rate = result.get('page_failure_rate', '0.00%')
                
                # Calculate success rate as percentage of successful pages
                total_pages = completed_pages + failed_pages
                if total_pages > 0:
                    success_rate = f"{(completed_pages / total_pages * 100):.2f}%"
                else:
                    success_rate = "0.00%"
                
                # is_successful is True only if ALL pages were processed successfully
                is_successful = failed_pages == 0 and completed_pages > 0
                
                # Simple completion print
                print(f"Successful pages: {completed_pages}, Failed pages: {failed_pages}, Success rate: {success_rate}")
                
                # Create success record
                success_record = ProcessRecord(
                    pdf_name=filename,
                    is_successful=is_successful,
                    num_successful_pages=completed_pages,
                    num_failed_pages=failed_pages,
                    success_rate=success_rate,
                    elapsed_time=elapsed_time,
                    start_time=start_time.strftime("%Y-%m-%d %H:%M:%S"),
                    finish_time=finish_time.strftime("%Y-%m-%d %H:%M:%S")
                )
                self.reporter.add_record(success_record)
                
                # Save output if requested
                if save_output and result.get('text'):
                    if not output_path:
                        output_path = f"{Path(file_path).stem}_extracted.md"
                    
                    try:
                        with open(output_path, "w", encoding="utf-8") as f:
                            f.write(result["text"])
                        print(f"Text saved to: {output_path}")
                    except IOError as e:
                        print(f"Warning: Could not save text file: {e}")
                
                return result
            else:
                error_msg = f"HTTP {response.status_code}: {response.text[:200]}"
                print(f"Processing failed: {error_msg}")
                
                # Create error record
                error_record = ProcessRecord(
                    pdf_name=filename,
                    is_successful=False,
                    num_successful_pages=0,
                    num_failed_pages=0,
                    success_rate="0.00%",
                    elapsed_time=elapsed_time,
                    start_time=start_time.strftime("%Y-%m-%d %H:%M:%S"),
                    finish_time=finish_time.strftime("%Y-%m-%d %H:%M:%S"),
                    error_message=error_msg
                )
                self.reporter.add_record(error_record)
                
                return {"error": f"HTTP {response.status_code}", "details": response.text}

        except requests.exceptions.Timeout:
            finish_time = datetime.now()
            elapsed_time = (finish_time - start_time).total_seconds()
            error_msg = "Request timed out after 2000 seconds"
            print(f"Processing failed: {error_msg}")
            
            # Create timeout error record
            error_record = ProcessRecord(
                pdf_name=filename,
                is_successful=False,
                num_successful_pages=0,
                num_failed_pages=0,
                success_rate="0.00%",
                elapsed_time=elapsed_time,
                start_time=start_time.strftime("%Y-%m-%d %H:%M:%S"),
                finish_time=finish_time.strftime("%Y-%m-%d %H:%M:%S"),
                error_message=error_msg
            )
            self.reporter.add_record(error_record)
            
            return {"error": "Timeout", "details": error_msg}
        except requests.exceptions.RequestException as e:
            finish_time = datetime.now()
            elapsed_time = (finish_time - start_time).total_seconds()
            error_msg = f"Request exception: {str(e)}"
            print(f"Processing failed: {error_msg}")
            
            # Create request error record
            error_record = ProcessRecord(
                pdf_name=filename,
                is_successful=False,
                num_successful_pages=0,
                num_failed_pages=0,
                success_rate="0.00%",
                elapsed_time=elapsed_time,
                start_time=start_time.strftime("%Y-%m-%d %H:%M:%S"),
                finish_time=finish_time.strftime("%Y-%m-%d %H:%M:%S"),
                error_message=error_msg
            )
            self.reporter.add_record(error_record)
            
            return {"error": "RequestException", "details": str(e)}

    def process_and_save(self, file_path: str, output_file_path: str) -> Dict[str, Any]:
        """
        Processes a document and saves the output directly to a file.
        
        Returns:
            Dict[str, Any]: The processing result from the server
        """
        result = self.process_document(file_path, save_output=False)  # We'll handle saving manually

        if "error" in result or "text" not in result:
            return result

        try:
            with open(output_file_path, "w", encoding="utf-8") as f:
                f.write(result["text"])
            print(f"Text saved to: {output_file_path}")
            return result
        except IOError as e:
            print(f"Error saving file: {e}")
            error_result = {"error": "File save failed", "details": str(e)}
            return {**result, **error_result}

def example_usage():
    """
    Example usage with the updated client
    """
    # Initialize client with custom JSON report file
    client = OlmOcrClient(
        base_url="https://8000-01k4t3gxyr35s21y2xjhx28e3e.cloudspaces.litng.ai",
        json_report_file="my_ocr_reports.json"
    )

    # Example PDF files
    pdf_files = os.listdir("H://python_projects//scientific//langextract_pdt_data_extraction//data//")
    base_folder= "H://python_projects//scientific//langextract_pdt_data_extraction//data//"
    save_folder = "H://python_projects//scientific//langextract_pdt_data_extraction//data//results//"

    print("="*70)
    
    for pdf in pdf_files:
        pdf_file = base_folder + pdf
        md_file = save_folder + pdf
        if not Path(pdf_file).exists():
            print(f"File not found: '{pdf_file}'. Skipping...")
            continue

        # Process the same document multiple times to show tracking
        for i in range(2):
            print(f"\n--- Processing {pdf_file} (attempt {i+1}) ---")
            result = client.process_document(
                file_path=pdf_file,
                save_output=True,
                output_path=f"{Path(pdf_file).stem}_attempt_{i+1}.md"
            )

    print(f"\n--- All processing attempts completed ---")

if __name__ == "__main__":
    example_usage()