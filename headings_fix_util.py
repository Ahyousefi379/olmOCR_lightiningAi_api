import re

def normalize_ocr(text: str) -> str:
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
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


def detect_author_block(text: str):
    """
    Return index of author/affiliation block if detected, else None.
    """
    paragraphs = split_paragraphs(text)
    for i, p in enumerate(paragraphs):
        if (
            re.search(r"@|e-?mail", p, re.I)  # email
            or re.search(r"department|university|institute", p, re.I)  # affiliation
            or re.search(r"[A-Z]\.\s*[A-Z][a-z]", p)  # initials like J. Smith
        ):
            return i
    return None


def insert_missing_abstract_intro(text: str) -> str:
    """
    Infer Abstract and Introduction if missing.
    """
    paragraphs = split_paragraphs(text)

    has_abs = re.search(r"(?im)^##\s*abstract", text)
    has_intro = re.search(r"(?im)^##\s*introduction", text)
    author_idx = detect_author_block(text)

    if has_abs and has_intro:
        return text  # nothing to fix

    new_blocks = []
    used = set()

    # Case A: Abstract before authors
    if not has_abs and author_idx and author_idx > 0:
        new_blocks.append("## Abstract\n\n" + paragraphs[0])
        used.add(0)

    # Case B: Abstract after authors
    elif not has_abs and author_idx is not None and author_idx < len(paragraphs) - 1:
        new_blocks.append("## Abstract\n\n" + paragraphs[author_idx + 1])
        used.add(author_idx + 1)

    # Case C: Fallback if no clue
    elif not has_abs:
        if len(paragraphs) >= 2:
            new_blocks.append("## Abstract\n\n" + paragraphs[0])
            new_blocks.append("## Introduction\n\n" + paragraphs[1])
            used.update([0, 1])
        elif len(paragraphs) == 1:
            new_blocks.append("## Abstract\n\n" + paragraphs[0])
            new_blocks.append("## Introduction\n\n" + paragraphs[0])
            used.add(0)

    # Insert Introduction if missing but abstract exists
    if not has_intro and has_abs:
        # Introduction starts after abstract
        abs_pos = next(
            (i for i, p in enumerate(paragraphs) if re.match(r"##\s*abstract", p, re.I)),
            None,
        )
        if abs_pos is not None and abs_pos + 1 < len(paragraphs):
            new_blocks.append("## Introduction\n\n" + paragraphs[abs_pos + 1])
            used.add(abs_pos + 1)

    # Rebuild
    result = []
    for i, p in enumerate(paragraphs):
        if i in used:
            continue
        result.append(p)

    # Prepend inferred blocks
    return "\n\n".join(new_blocks + result)


def convert_to_markdown(text: str) -> str:
    text = normalize_ocr(text)

    section_patterns = {
        "Abstract":      r"abstract",
        "Introduction":  r"[il1]ntroduction",
        "Background":    r"background",
        "Methods":       r"(methods?|materials\s+and\s+methods|experimental)",
        "Results":       r"results?",
        "Discussion":    r"discussion",
        "Results and Discussion": r"results?\s+(?:and\s+)?discussion",
        "Conclusions":   r"conclusions?|conclus[i1]ons?",
        "Summary":       r"summary",
        "References":    r"references?",
        "Acknowledgments": r"acknowledg(e)?ments?",
        "Appendix":      r"appendix",
        "Supporting Information": r"(supporting\s+information|supplementary\s+materials?)",
    }

    section_pattern = re.compile(
        r"^\s*[*_]*"
        r"(?:\d+(\.\d+)*\s*\.?\s*)?"
        r"(" + "|".join(section_patterns.values()) + r")"
        r"[*_]*"
        r"\s*:?\s*$",
        re.IGNORECASE | re.MULTILINE
    )

    def replacer(match):
        heading_word = match.group(2).strip()
        heading = heading_word[0].upper() + heading_word[1:].lower()
        return f"\n\n## {heading}\n"

    converted_text = section_pattern.sub(replacer, text)

    # Inline "ABSTRACT: ..." style
    converted_text = re.sub(
        r"(?im)^(abstract|introduction|methods?|results?|discussion|conclusions?|references?)\s*:\s*(.+)$",
        lambda m: f"\n\n## {m.group(1).capitalize()}\n\n{m.group(2)}",
        converted_text,
    )

    # Fix multi-line headings like "RESULTS AND\nDISCUSSION"
    converted_text = re.sub(
        r"(?im)^(results?\s+and)\s*$\n^(discussion)\s*$",
        lambda m: f"\n\n## {m.group(1).capitalize()} {m.group(2).capitalize()}\n",
        converted_text,
    )

    # Add missing abstract/intro intelligently
    converted_text = insert_missing_abstract_intro(converted_text)

    return converted_text
