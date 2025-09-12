import difflib

def merge_missing_content(base_text: str, secondary_text: str) -> str:
    """
    (This is the same function as before)
    Merges two texts by inserting missing lines from the secondary text into the base text.
    """
    base_lines = base_text.splitlines()
    secondary_lines = secondary_text.splitlines()
    s = difflib.SequenceMatcher(None, base_lines, secondary_lines, autojunk=False)
    merged_lines = []
    for tag, i1, i2, j1, j2 in s.get_opcodes():
        if tag in ('equal', 'delete', 'replace'):
            merged_lines.extend(base_lines[i1:i2])
        elif tag == 'insert':
            merged_lines.extend(secondary_lines[j1:j2])
    return "\n".join(merged_lines)

def symmetrical_merge(text1: str, text2: str) -> str:
    """
    Performs a two-way merge to get the most complete text by combining unique
    lines from both documents.

    Args:
        text1: The first text version.
        text2: The second text version.

    Returns:
        The most complete merged text.
    """
    # Run the merge with text1 as the base
    merged1 = merge_missing_content(base_text=text1, secondary_text=text2)
    
    # Run the merge with text2 as the base
    merged2 = merge_missing_content(base_text=text2, secondary_text=text1)
    
    # Return the version that has more content
    if len(merged1) >= len(merged2):
        return merged1
    else:
        return merged2

# Example usage
# -----------------------------
if __name__ == "__main__":
    with open("""H://python_projects//scientific//olmOCR//pdfs//raw ocr//Host–Guest Interactions in a Metal–Organic Framework Isoreticular Series for Molecular Photocatalytic CO2 Reduction._attempt 1_.md//_cleaned_Host–Guest Interactions in a Metal–Organic Framework Isoreticular Series for Molecular Photocatalytic CO2 Reduction_extracted.md""","r",encoding="utf8") as f:
        ocr1 = f.read()
    with open("""H://python_projects//scientific//olmOCR//pdfs//raw ocr//Host–Guest Interactions in a Metal–Organic Framework Isoreticular Series for Molecular Photocatalytic CO2 Reduction._attempt 2_.md//_cleaned_Host–Guest Interactions in a Metal–Organic Framework Isoreticular Series for Molecular Photocatalytic CO2 Reduction_extracted.md""","r",encoding="utf8") as f:
        ocr2 = f.read()


    merged = symmetrical_merge(ocr1, ocr2)
    with open("merged.md","w",encoding="utf8") as f:
        f.write(merged)
    print("Merged and normalized text:\n")
    #print(merged)
