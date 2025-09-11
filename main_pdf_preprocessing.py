from pdf_references_cleaner import remove_pages_after_references
from pathlib import Path
from olmOCR_api_util import OlmOcrPollingClient
import os

def main ():
    # directories
    base_dir = ""
    raw_pdfs_dir = f"{base_dir}//raw//"     # raw pdfs
    cleaned_pdfs_dir = f"{base_dir}//cleaned//"     # pdfs after removing references
    raw_ocr_dir =  f"{base_dir}//raw ocr//"     # raw text of pdf after OCR
    md_dir =  f"{base_dir}//md//"       # final processed text of pdfs 


    # get list of all raw pdf files
    raw_pdf_files = [f for f in os.listdir(raw_pdfs_dir) if f.lower().endswith('.pdf')]

    if not raw_pdf_files:
        print(f"No PDF files found in '{raw_pdfs_dir}'. Please add some PDFs to test.")
        return


    # removing reference pages
    for pdf in raw_pdf_files:
        remove_pages_after_references(input_pdf_path=f"{raw_pdfs_dir}//{raw_pdf_files}",
                                        output_pdf_path=f"{cleaned_pdfs_dir}//_cleaned_{raw_pdf_files}")



    # applying OCR on cleaned pdfs
    cleaned_pdf_files = [f for f in os.listdir(cleaned_pdfs_dir) if f.lower().endswith('.pdf')]

    SERVER_URL = "https://8000-01k4t3gxyr35s21y2xjhx28e3e.cloudspaces.litng.ai"
    POLL_INTERVAL_SECONDS= 20 #seconds
    TOTAL_WAIT_MINUTES= 60 #minutes
    OCR_ATTEMPTS = 2

    client = OlmOcrPollingClient(base_url=SERVER_URL,
                                 json_report_file=f"{raw_ocr_dir}//ocr_report.json")

    # running OCR
    for pdf_filename in cleaned_pdf_files:
        for i in range(OCR_ATTEMPTS):
            client.process_document(file_path=f"{cleaned_pdfs_dir}{pdf_filename}",
                                    save_output=True,
                                    output_dir=f"{raw_ocr_dir}{pdf_filename[:-3]}_attempt {i+1}_.md",
                                    poll_interval_sec=POLL_INTERVAL_SECONDS,
                                    total_wait_min=TOTAL_WAIT_MINUTES)
    



if __name__ == "__main__":
    main()