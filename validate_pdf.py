from PyPDF2 import PdfReader  
from pathlib import Path
from typing import List, Dict, Tuple
from openpyxl import Workbook  # <-- add this import


def validate_pdfs(source_dir: Path, save_excel_dir: Path) -> Tuple[List[Path], List[Path]]:
    """
    Validate all PDFs in the given directory.
    Return (valid_pdfs, invalid_pdfs).
    Also save invalid PDFs into an Excel file.
    """
    print(f"Validating PDFs in '{source_dir}'...")
    pdf_files = list(source_dir.glob('_cleaned_*.pdf'))

    if not pdf_files:
        print(f"No cleaned PDFs found in '{source_dir}'.")
        return [], []

    valid_pdfs = []
    invalid_pdfs = []

    for pdf_path in pdf_files:
        try:
            reader = PdfReader(str(pdf_path))
            _ = len(reader.pages)  # forces parsing
            valid_pdfs.append(pdf_path)
            print(f"  ✓ {pdf_path.name} is valid")
        except Exception as e:
            invalid_pdfs.append((pdf_path.name, str(e)))
            print(f"  ✗ {pdf_path.name} is INVALID ({e})")

    # --- Save invalid PDFs to Excel ---
    if invalid_pdfs:
        excel_path = save_excel_dir / "invalid_pdfs.xlsx"
        wb = Workbook()
        ws = wb.active
        ws.title = "Invalid PDFs"
        ws.append(["Filename", "Error"])  # header row
        for name, error in invalid_pdfs:
            ws.append([name, error])
        wb.save(str(excel_path))
        print(f"\nInvalid PDF list saved to {excel_path}")

    print(f"\nValidation complete. {len(valid_pdfs)} valid, {len(invalid_pdfs)} invalid.")
    if invalid_pdfs:
        print("Invalid PDFs:")
        for name, _ in invalid_pdfs:
            print(f"  - {name}")

    return valid_pdfs
