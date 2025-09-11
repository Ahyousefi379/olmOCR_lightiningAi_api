# PDF Reference Remover
#
# This script defines a Python function to read a PDF file, identify the page
# where the references section begins by detecting citation patterns (e.g., "[1]", "2."),
# and then create a new PDF file that excludes all pages that follow.
#
# This updated version uses the PyMuPDF library (fitz) for more reliable text extraction.
#
# Required library: PyMuPDF
# You can install it using pip:
# pip install PyMuPDF

import re
import fitz  # PyMuPDF

def remove_pages_after_references(input_pdf_path: str, output_pdf_path: str):
    """
    Reads a PDF, finds the page where references start based on citation
    patterns, and saves a new PDF without the reference pages and any pages
    that come after them. This is done by searching backwards from the end of the PDF.

    Args:
        input_pdf_path (str): The file path of the source PDF.
        output_pdf_path (str): The file path where the modified PDF will be saved.
    """
    try:
        # 1. Initialize a PDF reader object using PyMuPDF
        doc = fitz.open(input_pdf_path)
        num_pages = len(doc)
        print(f"Successfully opened '{input_pdf_path}' with {num_pages} pages.")

        # 2. Find the page where references begin by searching backwards from the end.
        last_page_to_keep = -1
        
        # Define a regex pattern to find common citation formats like "[1]" or "1."
        # at the beginning of a line.
        # (?:^\[\d+\]) matches a pattern like "[12]" at the start of a line.
        # (?:^\s*\d+\.) matches a pattern like " 1." or "12." at the start of a line.
        pattern = re.compile(r'(?:^\s*\[\d+\])|(?:^\s*\d+\.)', re.MULTILINE)

        # Iterate from the last page backwards to the first page
        for i in range(num_pages - 1, -1, -1):
            page = doc.load_page(i)
            text = page.get_text()
            
            is_reference_page = False
            if text:
                matches = pattern.findall(text)
                # If we find a high number of reference-style entries, it's a reference page.
                # A threshold of 5 helps avoid false positives.
                if len(matches) > 5:
                    is_reference_page = True
            
            # If we find the first page from the end that IS NOT a reference page,
            # that's the last page of the main content we want to keep.
            if not is_reference_page:
                last_page_to_keep = i+1
                print(f"Content appears to end on page {i + 1}. Pages after this will be removed.")
                break
        
        # Determine how many pages to write to the new file
        if last_page_to_keep == -1:
            print("The entire document appears to be a reference list. The output file will be empty.")
            pages_to_write = 0
        elif last_page_to_keep == num_pages:
            print("No reference section found at the end. The output file will be a copy of the original.")
            pages_to_write = num_pages
        else:
            pages_to_write = last_page_to_keep + 1

        # 3. Create a new PDF with pages up to the last content page
        if pages_to_write > 0:
            # Select the pages to keep (from page 0 up to last_page_to_keep)
            doc.select(range(pages_to_write))
            # Save the modified document
            doc.save(output_pdf_path)
            print(f"Successfully created '{output_pdf_path}' with {pages_to_write} pages.")
        else:
            # Create an empty PDF if no pages are to be kept
            new_doc = fitz.open()
            new_doc.save(output_pdf_path)
            print(f"The output file '{output_pdf_path}' is empty as requested.")
        
        doc.close()

    except FileNotFoundError:
        print(f"Error: The file '{input_pdf_path}' was not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


# --- Example Usage ---
if __name__ == '__main__':
    # To use this script:
    # 1. Make sure you have a PDF file you want to process.
    # 2. Change 'input_document.pdf' to the path of your PDF file.
    # 3. Change 'output_document_trimmed.pdf' to your desired output file name.
    
    # NOTE: You must have a PDF file named 'input_document.pdf' in the same
    # directory as this script for this example to work out-of-the-box.
    
    input_pdf = 'H://python_projects//scientific//langextract_pdt_data_extraction//data//Fe3O4‐Incorporated Metal‐Organic Framework for Chemo_Ferroptosis Synergistic Anti‐Tumor via the Enhanced Chemodynamic Therapy.pdf'
    output_pdf = 'H://python_projects//scientific//langextract_pdt_data_extraction//data//fixed Fe3O4‐Incorporated Metal‐Organic Framework for Chemo_Ferroptosis Synergistic Anti‐Tumor via the Enhanced Chemodynamic Therapy.pdf'

    # Create a dummy PDF for testing if it doesn't exist
    print("-" * 20)
    print("Processing PDF...")
    remove_pages_after_references(input_pdf, output_pdf)
    print("-" * 20)

