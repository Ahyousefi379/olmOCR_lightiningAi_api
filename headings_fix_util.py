import re

def normalize_ocr(text: str) -> str:
    """Fix common OCR artifacts."""
    replacements = {
        "ﬁ": "fi", "ﬂ": "fl",
        "‘": "'", "’": "'",
        "“": '"', "”": '"',
        "—": "-", "–": "-",
        "¬": "-",
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    return text

def split_paragraphs(text: str):
    """Split text into paragraphs by empty lines."""
    return [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]

def is_author_or_metadata_line(paragraph: str) -> bool:
    """Detect authors, affiliations, emails, ARTICLE INFO, keywords, etc."""
    return bool(
        re.search(r"@|e-?mail", paragraph, re.I) or
        re.search(r"department|university|institute", paragraph, re.I) or
        re.search(r"[A-Z]\.\s*[A-Z][a-z]", paragraph) or
        re.search(r"\*|a,|b,|c,", paragraph) or
        re.search(r"ARTICLE\s+INFO|Keywords", paragraph, re.I)
    )

def detect_author_block(paragraphs):
    """Return index of author/metadata block, if any."""
    for i, p in enumerate(paragraphs):
        if is_author_or_metadata_line(p):
            return i
    return None

def detect_explicit_heading(paragraphs, heading_keywords):
    """Return the index of explicit heading, if found."""
    for i, p in enumerate(paragraphs):
        for kw in heading_keywords:
            # Handle A B S T R A C T spacing
            pattern = r"^\s*" + r"\s*".join(list(kw.lower())) + r"\s*:?\s*$"
            if re.match(pattern, p.lower()):
                return i
        # Normal heading detection
        if re.match(rf"^{kw}\s*:?", p, re.I):
            return i
    return None

def convert_to_markdown(text: str) -> str:
    text = normalize_ocr(text)
    paragraphs = split_paragraphs(text)

    # ---- Detect Title ----
    title_idx = 0 if len(paragraphs[0].split()) < 15 else None

    # ---- Detect Author/Metadata Block ----
    author_idx = detect_author_block(paragraphs)

    # ---- Detect Abstract ----
    abstract_keywords = ["abstract"]
    abs_idx = detect_explicit_heading(paragraphs, abstract_keywords)

    # Insert Abstract if missing
    if abs_idx is None:
        if title_idx is not None and author_idx and title_idx < author_idx:
            # Paragraphs before authors
            abs_idx = title_idx + 1
            paragraphs.insert(abs_idx, "## Abstract\n\n" + paragraphs[abs_idx])
        elif author_idx is not None and author_idx < len(paragraphs) - 1:
            # Paragraphs after author block
            abs_idx = author_idx + 1
            paragraphs.insert(abs_idx, "## Abstract\n\n" + paragraphs[abs_idx])
        else:
            # Fallback: first paragraph > 5 words
            for i, p in enumerate(paragraphs):
                if len(p.split()) > 5:
                    abs_idx = i
                    paragraphs.insert(abs_idx, "## Abstract\n\n" + paragraphs[abs_idx])
                    break

    # ---- Detect Introduction ----
    intro_keywords = ["introduction"]
    intro_idx = detect_explicit_heading(paragraphs, intro_keywords)

    # Insert Introduction if missing
    if intro_idx is None and abs_idx is not None:
        # Paragraph immediately after Abstract
        next_idx = abs_idx + 1
        if next_idx < len(paragraphs):
            # Duplicate if Abstract paragraph == Introduction paragraph
            intro_text = paragraphs[next_idx] if next_idx != abs_idx else paragraphs[abs_idx]
            paragraphs.insert(next_idx, "## Introduction\n\n" + intro_text)

    # ---- Normalize other common headings ----
    other_headings = ["Methods","Materials and Methods","Experimental","Results","Discussion",
                      "Results and Discussion","Conclusions","Summary","References",
                      "Acknowledgments","Appendix","Supporting Information"]

    heading_pattern = re.compile(
        r"^\s*[*_]*"                       # optional bold/italic
        r"(?:\d+(\.\d+)*\s*\.?\s*)?"       # optional numbering
        r"(" + "|".join([h.lower() for h in other_headings]) + r")"
        r"[*_]*"
        r"\s*:?\s*$",
        re.IGNORECASE | re.MULTILINE
    )

    for i, p in enumerate(paragraphs):
        if heading_pattern.match(p):
            paragraphs[i] = "## " + p.strip().capitalize()

    return "\n\n".join(paragraphs)

# ---- Example Usage ----
if __name__ == "__main__":

    md_text = convert_to_markdown()
    print(md_text)
