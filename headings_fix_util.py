from pathlib import Path
import pandas as pd
import re
import string
import time


class ScientificTextToMarkdownConverter:
    """
    Converts OCR'd scientific paper text into a structured Markdown file.
    Integrates with Excel file to prepend abstract and mark processing status.

    The process is as follows:
    1.  Load the OCR text from filepath
    2.  Search Excel file for matching paper name
    3.  Prepend the abstract from Excel to the OCR text
    4.  Mark the paper as processed (True) in Excel
    5.  Return the processed text with abstract prepended
    """

    def __init__(self, ocr_filepath: Path, excel_filepath: Path, log_file: Path = None):
        """
        Initialize converter with OCR file and Excel database.
        
        Args:
            ocr_filepath: Path to the OCR'd text file
            excel_filepath: Path to the Excel file containing paper information
            log_file: Optional path to save detailed matching logs
        """
        self.ocr_filepath = Path(ocr_filepath)
        self.excel_filepath = Path(excel_filepath)
        self.log_file = log_file
        self.match_found = False
        self.processed_text = None
        self.log_entries = []
        
        # Load OCR text
        with open(self.ocr_filepath, 'r', encoding='utf-8') as f:
            ocr_text = f.read()
        
        # Process with Excel integration
        self.processed_text = self._process_with_excel(ocr_text)
        
        # Write log if log_file specified
        if self.log_file and self.log_entries:
            self._write_log()
    
    def _log(self, message: str):
        """Add a message to the log entries."""
        self.log_entries.append(message)
        print(message)
    
    def _write_log(self):
        """Write accumulated log entries to file."""
        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write('\n'.join(self.log_entries) + '\n\n')
        except Exception as e:
            print(f"    Warning: Could not write to log file: {e}")

    def _create_comparison_string(self, text: str) -> str:
        """
        Create a normalized string for comparison by removing all special characters
        and extra whitespace, keeping only alphanumeric characters and spaces.
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove all special characters except spaces
        allowed = string.ascii_lowercase + string.digits + ' '
        text = ''.join(c if c in allowed else ' ' for c in text)
        
        # Normalize whitespace
        text = ' '.join(text.split())
        
        return text
    
    def _find_best_match(self, df: pd.DataFrame, paper_name: str) -> int:
        """
        Find the best matching row for the paper name in the DataFrame.
        Handles cases where filename differs from title due to invalid Windows characters.
        Uses aggressive normalization to match titles with replaced special characters.
        
        Returns:
            Index of the matching row, or -1 if no match found
        """
        # Create comparison version of paper name
        normalized_paper_name = self._create_comparison_string(paper_name)
        
        self._log(f"    Searching for: '{paper_name}'")
        self._log(f"    Normalized to: '{normalized_paper_name}'")
        
        # First pass: Try exact match after aggressive normalization
        for idx, title in enumerate(df['Title'].astype(str)):
            if title == 'nan':
                continue
            normalized_title = self._create_comparison_string(title)
            if normalized_paper_name == normalized_title:
                self._log(f"    ✓ Exact match found!")
                return df.index[idx]
        
        self._log(f"    No exact match, trying fuzzy matching...")
        
        # Second pass: Try very close match (allows minor differences)
        paper_words = set(normalized_paper_name.split())
        
        best_match_idx = -1
        best_similarity = 0.0
        best_title = ""
        
        for idx, title in enumerate(df['Title'].astype(str)):
            if title == 'nan':
                continue
            normalized_title = self._create_comparison_string(title)
            title_words = set(normalized_title.split())
            
            if not paper_words or not title_words:
                continue
            
            # Calculate Jaccard similarity
            intersection = len(paper_words & title_words)
            union = len(paper_words | title_words)
            similarity = intersection / union if union > 0 else 0
            
            # Also check if one is contained in the other (for subset matches)
            if len(paper_words) > 0:
                containment = intersection / len(paper_words)
            else:
                containment = 0
            
            # Use the higher of the two metrics
            final_similarity = max(similarity, containment)
            
            if final_similarity > best_similarity:
                best_similarity = final_similarity
                best_match_idx = df.index[idx]
                best_title = title
        
        if best_match_idx != -1 and best_similarity >= 0.85:
            self._log(f"    Fuzzy match found (similarity: {best_similarity:.2%})")
            self._log(f"    Matched to: '{best_title}'")
            return best_match_idx
        
        if best_match_idx != -1:
            self._log(f"    Best match was '{best_title}' but similarity too low: {best_similarity:.2%}")
        
        return -1

    def _process_with_excel(self, ocr_text: str) -> str:
        """
        Search Excel for paper, prepend abstract, and mark as processed.
        
        Args:
            ocr_text: The original OCR text
            
        Returns:
            Text with abstract prepended as markdown heading, or original text if no match
        """
        # Load Excel file with retry logic for file access
        max_retries = 3
        df = None
        for attempt in range(max_retries):
            try:
                df = pd.read_excel(self.excel_filepath)
                break
            except PermissionError:
                if attempt == max_retries - 1:
                    self._log(f"    ✗ Error: Cannot access Excel file. Please close it in Excel and try again.")
                    self.match_found = False
                    return ocr_text
                time.sleep(1)
        
        if df is None:
            self._log(f"    ✗ Error: Could not load Excel file.")
            self.match_found = False
            return ocr_text
        
        # Verify required columns exist
        if 'Title' not in df.columns:
            self._log("    ✗ Error: 'Title' column not found in Excel file.")
            self.match_found = False
            return ocr_text
        if 'Abstract' not in df.columns:
            self._log("    ✗ Error: 'Abstract' column not found in Excel file.")
            self.match_found = False
            return ocr_text
        if 'Processed' not in df.columns:
            self._log("    Warning: 'Processed' column not found. Creating it.")
            df['Processed'] = False
        else:
            # Fix data type of Processed column if it's not boolean
            # Convert to object type first to avoid the warning, then to bool
            if df['Processed'].dtype != 'bool':
                df['Processed'] = df['Processed'].fillna(False).astype('object')
        
        # Extract paper name from OCR filename (without extension)
        paper_name = self.ocr_filepath.stem
        
        # Remove _attempt_X suffix if present (from OCR processing)
        paper_name = re.sub(r'_attempt_?\d+$', '', paper_name)
        
        # Find matching row
        row_idx = self._find_best_match(df, paper_name)
        
        if row_idx == -1:
            self._log(f"    ✗ No match found in Excel")
            self.match_found = False
            return ocr_text
        
        self.match_found = True
        self._log(f"    ✓ Found match: '{df.loc[row_idx, 'Title']}'")
        
        # Get abstract from Excel
        abstract_text = df.loc[row_idx, 'Abstract']
        
        # Handle empty or NaN abstracts
        if pd.isna(abstract_text) or str(abstract_text).strip() == '' or str(abstract_text) == 'nan':
            abstract_text = ""
            self._log("    Note: Abstract is empty for this paper.")
        else:
            abstract_text = str(abstract_text).strip()
        
        # Mark as processed - now safe because column is object type
        df.at[row_idx, 'Processed'] = True
        
        # Save Excel file with retry logic
        for attempt in range(max_retries):
            try:
                df.to_excel(self.excel_filepath, index=False)
                self._log(f"    ✓ Marked as processed in Excel")
                break
            except PermissionError:
                if attempt == max_retries - 1:
                    self._log(f"    ✗ Warning: Could not save Excel file (file is open). Match found but Excel not updated.")
                else:
                    time.sleep(1)
        
        # Prepend abstract as markdown heading
        if abstract_text:
            return f"# Abstract\n\n{abstract_text}\n\n{ocr_text}"
        else:
            return ocr_text

    def get_text(self) -> str:
        """
        Returns the processed text with abstract prepended.
        
        Returns:
            The processed text
        """
        return self.processed_text
    
    def save_to_file(self, output_path: Path) -> bool:
        """
        Save the processed text to a file, but ONLY if a match was found in Excel.
        
        Args:
            output_path: Path where the file should be saved
            
        Returns:
            True if file was saved, False if no match was found (file not created)
        """
        if not self.match_found:
            self._log(f"    ✗ Skipping file creation (no Excel match)")
            return False
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(self.processed_text)
        
        self._log(f"    ✓ Saved to '{output_path.name}'")
        return True


