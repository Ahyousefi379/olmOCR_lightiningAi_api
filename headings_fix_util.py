#!/usr/bin/env python3
"""
smart_md_converter.py - Improved version with better duplicate removal and section detection

Take noisy OCR text of a scientific paper and produce a lightweight Markdown
where main section headings (Abstract, Introduction, Methods, Results, etc.)
are normalized to "## Heading". The converter preserves main body text,
figures/tables/captions, and attempts to intelligently insert missing Abstract
and Introduction sections.

Usage: call process_paper(text, metadata_action="extract") -> markdown_text

metadata_action: "extract" (default) => collects metadata and places it at the top
                 "keep"    => do nothing (leave metadata in place)
                 "remove"  => drop metadata paragraphs (not recommended)
"""

import re
from typing import List, Optional, Tuple, Dict
from difflib import SequenceMatcher

# -------------------------
# Configuration thresholds
# -------------------------
TITLE_MAX_WORDS = 30       # max words to consider the first paragraph a title
TITLE_MIN_WORDS = 5          # minimum words for a title (avoid weird blanks)
TITLE_PROTECT_WORDS = 100      # headings shorter than this won't be treated as abstract
ABSTRACT_MIN_WORDS = 100      # min words for a paragraph to be considered abstract-ish
FALLBACK_DUPLICATE_THRESHOLD = 2  # paragraphs needed to do fallback duplicate
SIMILARITY_THRESHOLD = 0.85  # For detecting duplicate lines

# -------------------------
# Utilities & normalization
# -------------------------
def normalize_ocr(text: str) -> str:
    """Fix common OCR issues and normalize common glyphs."""
    repl = {
        "ﬁ": "fi", "ﬂ": "fl",
        "'": "'", "'": "'", """: '"', """: '"',
        "—": "-", "–": "-", "¬": "-",
        "\r\n": "\n", "\r": "\n",
    }
    for k, v in repl.items():
        text = text.replace(k, v)
    # Trim trailing spaces on each line
    text = "\n".join(line.rstrip() for line in text.splitlines())
    # Collapse >2 blank lines into two
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)
    return text


def split_paragraphs(text: str) -> List[str]:
    """Split text into paragraphs using blank lines as separators (keeps paragraphs intact)."""
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip() != ""]
    return paras


def join_paragraphs(paragraphs: List[str]) -> str:
    """Join paragraphs back into text using two newlines."""
    return "\n\n".join(paragraphs)


def words_count(s: str) -> int:
    return len(re.findall(r"\w+", s))


def normalize_for_heading_match(s: str) -> str:
    """
    Create a simplified version of s suitable for heading keyword matching:
    lowercased, letters & digits only, collapsed.
    E.g., "A B S T R A C T" -> "abstract".
    """
    return re.sub(r"[^a-z0-9]+", "", s.lower() or "")


# -------------------------
# Known sections (ordered: long ones first)
# -------------------------
SECTION_PATTERNS = [
    ("resultsanddiscussion", r"results\s*(?:and)?\s*discussion"),  # long first
    ("materialsandmethods", r"materials\s*and\s*methods"),
    ("supportinginformation", r"supporting\s*information|supplementary\s+materials?"),
    ("acknowledgments", r"acknowledg(e)?ments?"),
    ("references", r"references?"),
    ("conclusions", r"conclusions?|conclus[i1]ons?"),
    ("discussion", r"discussion"),
    ("results", r"results?"),
    ("methods", r"methods?|experimental"),
    ("background", r"background"),
    ("summary", r"summary"),
    ("appendix", r"appendix"),
    ("abstract", r"abstract"),
    ("introduction", r"[il1]ntroduction"),
]


def canonical_name_from_key(key: str) -> str:
    """Return user-friendly canonical title-casing for section keys."""
    mapping = {
        "resultsanddiscussion": "Results and Discussion",
        "materialsandmethods": "Materials and Methods",
        "supportinginformation": "Supporting Information",
        "acknowledgments": "Acknowledgments",
        "references": "References",
        "conclusions": "Conclusions",
        "discussion": "Discussion",
        "results": "Results",
        "methods": "Methods",
        "background": "Background",
        "summary": "Summary",
        "appendix": "Appendix",
        "abstract": "Abstract",
        "introduction": "Introduction",
    }
    return mapping.get(key, key.title())


# -------------------------
# Duplicate detection and removal
# -------------------------
def remove_duplicate_titles(paragraphs: List[str]) -> List[str]:
    """
    Remove duplicate title-like paragraphs at the beginning of the document.
    Common in OCR where title appears multiple times.
    """
    if len(paragraphs) < 2:
        return paragraphs
    
    # Check first few paragraphs for duplicates
    cleaned = []
    seen_titles = []
    
    for i, p in enumerate(paragraphs[:10]):  # Only check first 10 paragraphs
        # Skip if too short or too long to be a title
        if words_count(p) < 2:  # Very short, likely noise
            cleaned.append(p)
            continue
            
        # Check similarity with previously seen titles
        is_duplicate = False
        for seen in seen_titles:
            similarity = SequenceMatcher(None, p.lower().strip(), seen.lower().strip()).ratio()
            if similarity > SIMILARITY_THRESHOLD:
                is_duplicate = True
                break
        
        if not is_duplicate:
            cleaned.append(p)
            # Only track as potential title if it's in the first few positions and reasonable length
            if i < 5 and TITLE_MIN_WORDS <= words_count(p) <= TITLE_MAX_WORDS * 2:
                seen_titles.append(p)
        # If duplicate, we skip it (don't add to cleaned)
    
    # Add remaining paragraphs
    cleaned.extend(paragraphs[10:])
    return cleaned


# -------------------------
# Metadata extraction
# -------------------------
METADATA_LABELS = [
    r"^title\s*:",
    r"^authors?\s*:",
    r"^to\s+be\s+cited\s+as",
    r"^link\s+to\s+vor",
    r"^doi\s*:",
    r"accepted\s+article",
    r"voR",
    r"earl(y)?\s+view",
    r"^to\s+be\s+cited",
    r"^manuscript\s+has\s+been\s+accepted",
    r"manuscript\s+submitted",
    r"revised\s+manuscript\s+received",
    r"published\s+january",
    r"©.*published by ecs",
]


def extract_metadata(paragraphs: List[str]) -> Tuple[List[str], Optional[str]]:
    """
    Find any paragraphs that look like metadata (Title:, Authors:, DOI, 'Accepted Article', etc.)
    Return (remaining_paragraphs, metadata_text or None).
    """
    metadata = []
    remaining = []
    top_zone_limit = min(8, len(paragraphs))
    used_indices = set()

    # First pass: explicit label detection
    for i, p in enumerate(paragraphs[:top_zone_limit]):
        low = p.lower()
        if any(re.search(lbl, low) for lbl in METADATA_LABELS) or re.search(r"\bdoi\b|https?://doi.org|to be cited as", low):
            metadata.append(p)
            used_indices.add(i)

    # If no explicit metadata found, still try to catch a typical top area
    if not metadata:
        if len(paragraphs) >= 2:
            first, second = paragraphs[0], paragraphs[1]
            if 1 <= words_count(first) <= TITLE_MAX_WORDS and (re.search(r"\bdepartment\b|\buniversity\b|\binstitute\b|@|,|\*", second, re.I)):
                metadata.extend([first, second])
                used_indices.update({0, 1})

    # Build remaining list
    for i, p in enumerate(paragraphs):
        if i in used_indices:
            continue
        remaining.append(p)

    if metadata:
        return remaining, "\n\n".join(metadata)
    return paragraphs, None


# -------------------------
# Enhanced author detection
# -------------------------
def is_author_paragraph(p: str) -> bool:
    """Enhanced author/affiliation detection."""
    # Check for numbered affiliations (1Department, 2School, etc.)
    if re.search(r'^\d+\s*(Department|School|Institute|Center|Faculty|College)', p, re.I):
        return True
    
    # Check for superscript markers followed by institution names
    if re.search(r'^[a-z\d,\s]+\s+(Department|School|Institute|University|Center|Faculty|College)', p, re.I):
        return True
    
    # Check for author names with specific patterns
    if re.search(r'[A-Z][a-z]+\s+[A-Z]\.\s*[A-Z][a-z]+', p):  # "FirstName M. LastName"
        return True
    if re.search(r'[A-Z][a-z]+\s+[A-Z][a-z]+,\s+[A-Z][a-z]+\s+[A-Z][a-z]+', p):  # Multiple names with commas
        return True
    
    # Names with "and" pattern (common in author lists)
    if re.search(r'[A-Z][a-z]+\s+[A-Z][a-z]+.*\s+and\s+[A-Z][a-z]+\s+[A-Z]', p):
        return True
    
    # Check for affiliation indicators
    return bool(
        re.search(r"@|e-?mail", p, re.I) or
        re.search(r"\bdepartment\b|\buniversity\b|\binstitute\b|\bschool\b|\bcenter\b|\bcollege\b", p, re.I) or
        re.search(r"[A-Z]\.\s*[A-Z][a-z]", p) or
        re.search(r"\*|†|‡|§|¶|, a|, b|, c|, d", p) or
        re.search(r"^authors?\s*:", p, re.I) or
        re.search(r"^affiliations?\s*:", p, re.I) or
        re.search(r", (USA|UK|China|Germany|France|Japan|Korea|Canada|Australia|India)\b", p)
    )


def detect_author_block_range(paragraphs: List[str]) -> Optional[Tuple[int, int]]:
    """
    Return the range (start_idx, end_idx) of author/affiliation paragraphs.
    This helps identify multiple consecutive author/affiliation paragraphs.
    """
    start_idx = None
    end_idx = None
    
    for i, p in enumerate(paragraphs[:15]):  # Check first 15 paragraphs
        if is_author_paragraph(p):
            if start_idx is None:
                start_idx = i
            end_idx = i
        elif start_idx is not None and i > start_idx + 1:
            # Allow one non-author paragraph in between (for formatting issues)
            # but if we see two consecutive non-author paragraphs, stop
            if i > 0 and not is_author_paragraph(paragraphs[i-1]):
                break
    
    if start_idx is not None:
        return (start_idx, end_idx)
    return None


def detect_author_block_index(paragraphs: List[str]) -> Optional[int]:
    """Return the first index that looks like an author/affiliation paragraph; else None."""
    for i, p in enumerate(paragraphs[:10]):
        if is_author_paragraph(p):
            return i
    return None


# -------------------------
# Heading detection per paragraph
# -------------------------
def match_heading_paragraph(paragraph: str) -> Optional[Tuple[str, str]]:
    """
    If paragraph looks like a heading, return (canonical_name, rest_of_line_text).
    rest_of_line_text is the trailing text on the same paragraph after the colon (if any).
    Returns None if not matched.
    """
    p = paragraph.strip()

    # Remove bold/italic wrappers just for matching
    p_unwrapped = re.sub(r"^[*_]+\s*|\s+[*_]+$", "", p)

    # If paragraph is all-letters-with-spaces (A B S T R A C T) collapse for matching
    compact = normalize_for_heading_match(p_unwrapped)

    # Try direct compact equality
    for key, pattern in SECTION_PATTERNS:
        if compact == re.sub(r"[^a-z0-9]+", "", pattern.lower()):
            return canonical_name_from_key(key), ""

    # Regex: detect common heading forms
    for key, pat in SECTION_PATTERNS:
        regex = rf"^\s*[*_]*\s*(?:\d+(?:\.\d+)*[\.KATEX_INLINE_CLOSE]?\s*)?(?P<h>{pat})\s*[*_]*\s*:?\s*(?P<rest>.*)$"
        m = re.match(regex, p_unwrapped, flags=re.IGNORECASE)
        if m:
            rest = (m.group("rest") or "").strip()
            return canonical_name_from_key(key), rest

    return None


# -------------------------
# Enhanced abstract and introduction detection
# -------------------------
def looks_like_abstract(p: str) -> bool:
    """
    Enhanced heuristics to detect abstract-like paragraphs.
    """
    # Must be reasonably long
    if words_count(p) < ABSTRACT_MIN_WORDS:
        return False
    
    # Common abstract starting phrases
    abstract_starters = [
        r'^A (representative|novel|new|simple|facile|efficient)',
        r'^This (paper|work|study|article|report)',
        r'^We (report|present|demonstrate|show|describe|investigate)',
        r'^Here,?\s+we',
        r'^In this (paper|work|study|article)',
        r'^The (present|current) (work|study|paper)',
        r'^[A-Z][a-z]+\s+(is|are|was|were|has been|have been)\s+(synthesized|prepared|developed|investigated|studied)',
        r'^(Novel|New|Efficient|Facile)\s+[a-z]+',
        r'^[A-Z][a-z]+\s+[a-z]+\s+(nanoparticles?|compounds?|materials?|catalysts?|frameworks?)',
    ]
    
    for pattern in abstract_starters:
        if re.search(pattern, p, re.I):
            return True
    
    # Check for abstract-like content patterns
    abstract_keywords = [
        r'\b(synthesized?|prepared?|investigated?|studied|developed?|demonstrated?|characterized?)\b',
        r'\b(results?|show|reveal|indicate|suggest|demonstrate)\b',
        r'\b(novel|new|efficient|facile|simple|improved)\b.*\b(method|approach|strategy|technique)\b',
        r'\b(herein|in this work|we report)\b',
    ]
    
    keyword_count = sum(1 for pattern in abstract_keywords if re.search(pattern, p, re.I))
    if keyword_count >= 2:
        return True
    
    return False


def looks_like_introduction(p: str) -> bool:
    """
    Detect introduction-like paragraphs.
    """
    # Must be reasonably long
    if words_count(p) < 50:
        return False
    
    # Common introduction phrases
    intro_patterns = [
        r'is a compound of undeniable',
        r'(is|are|has been|have been) (widely|extensively) (used|studied|investigated|applied)',
        r'has (received|attracted|gained) (considerable|significant|much|increasing|growing)',
        r'in recent (years|decades)',
        r'over the (past|last) (decade|years|century)',
        r'plays? (a|an) (important|crucial|critical|key|vital|essential) role',
        r'fundamental (importance|understanding|interest)',
        r'commercial (significance|importance|application|interest)',
        r'(great|significant|considerable) (attention|interest) (has been|have been)',
        r'(However|Nevertheless|Despite|Although)',
        r'remains? (a )?(challenge|challenging|difficult)',
        r'(traditional|conventional|current) (methods?|approaches?|techniques?)',
    ]
    
    for pattern in intro_patterns:
        if re.search(pattern, p, re.I):
            return True
    
    return False


# -------------------------
# Broken multi-line heading joiner
# -------------------------
def join_broken_headings(paragraphs: List[str]) -> List[str]:
    """
    Detect pairs of adjacent paragraphs that are fragments of a heading and join them.
    E.g. "RESULTS AND" + "DISCUSSION" -> "Results and Discussion"
    """
    i = 0
    out = []
    while i < len(paragraphs):
        cur = paragraphs[i]
        if i + 1 < len(paragraphs):
            nxt = paragraphs[i + 1]
            if words_count(cur) <= 6 and words_count(nxt) <= 6:
                combined = cur.strip() + " " + nxt.strip()
                match = match_heading_paragraph(combined)
                if match:
                    canon, rest = match
                    heading_para = f"## {canon}"
                    out.append(heading_para)
                    if rest:
                        out.append(rest)
                    i += 2
                    continue
        out.append(cur)
        i += 1
    return out


# -------------------------
# Improved abstract and introduction insertion
# -------------------------
def insert_abstract_and_intro_improved(paragraphs: List[str]) -> List[str]:
    """
    Enhanced version with better heuristics for finding abstract and introduction.
    """
    paras = paragraphs[:]
    
    # First, check if headings already exist
    abs_index = None
    intro_index = None
    
    for i, p in enumerate(paras):
        if p.strip().lower().startswith("## abstract"):
            abs_index = i
        elif p.strip().lower().startswith("## introduction"):
            intro_index = i
    
    # If both exist, we're done
    if abs_index is not None and intro_index is not None:
        return paras
    
    # Find author block range
    author_range = detect_author_block_range(paras)
    
    # Strategy 1: Look for abstract after authors
    if abs_index is None and author_range:
        # Check paragraphs immediately after author block
        check_start = author_range[1] + 1
        for i in range(check_start, min(check_start + 3, len(paras))):
            if i < len(paras) and looks_like_abstract(paras[i]):
                paras.insert(i, "## Abstract")
                abs_index = i
                break
    
    # Strategy 2: Look for abstract before authors
    if abs_index is None and author_range and author_range[0] > 0:
        # Check paragraphs before author block
        for i in range(author_range[0] - 1, -1, -1):
            if looks_like_abstract(paras[i]):
                paras.insert(i, "## Abstract")
                abs_index = i
                break
    
    # Strategy 3: If no authors found, look in first few paragraphs
    if abs_index is None:
        for i in range(min(5, len(paras))):
            if looks_like_abstract(paras[i]):
                paras.insert(i, "## Abstract")
                abs_index = i
                break
    
    # Find introduction - can be further down in the document
    if intro_index is None:
        # Start looking after abstract (if found) or after authors
        start_search = 0
        if abs_index is not None:
            start_search = abs_index + 2  # Skip abstract heading and content
        elif author_range:
            start_search = author_range[1] + 1
        
        # Look through more of the document for introduction
        for i in range(start_search, min(len(paras), 20)):  # Check up to 20 paragraphs
            if looks_like_introduction(paras[i]):
                paras.insert(i, "## Introduction")
                intro_index = i
                break
    
    # Fallback: If still no abstract found, use first substantial paragraph
    if abs_index is None:
        for i, p in enumerate(paras[:10]):
            if words_count(p) >= ABSTRACT_MIN_WORDS:
                paras.insert(i, "## Abstract")
                abs_index = i
                break
    
    # Fallback: If still no introduction found, look for the next substantial paragraph after abstract
    if intro_index is None and abs_index is not None:
        # Find next substantial paragraph after abstract
        search_start = abs_index + 2 if abs_index is not None else 0
        for i in range(search_start, min(len(paras), search_start + 10)):
            if words_count(paras[i]) >= 100 and not paras[i].startswith("##"):
                paras.insert(i, "## Introduction")
                intro_index = i
                break
    
    # Last resort: insert empty Introduction after abstract if nothing found
    if intro_index is None and abs_index is not None:
        insert_pos = abs_index + 2
        if insert_pos <= len(paras):
            paras.insert(insert_pos, "## Introduction")
    
    return paras


# -------------------------
# Normalize other headings
# -------------------------
def normalize_other_headings(paragraphs: List[str]) -> List[str]:
    """
    Convert paragraphs that are explicit headings (Methods, Results, etc.) into "## <Canonical>".
    If heading paragraph contains inline rest-text, keep it as a following paragraph.
    """
    out = []
    for p in paragraphs:
        # Skip if already a markdown heading
        if p.strip().startswith("##"):
            out.append(p)
            continue
            
        m = match_heading_paragraph(p)
        if m:
            canon, rest = m
            out.append(f"## {canon}")
            if rest:
                out.append(rest)
        else:
            out.append(p)
    return out


# -------------------------
# Top-level processing pipeline
# -------------------------
def process_paper(text: str, metadata_action: str = "extract") -> str:
    """
    Main pipeline:
    1) Normalize OCR artifacts.
    2) Split into paragraphs.
    3) Remove duplicate title lines.
    4) Extract metadata (optional).
    5) Join broken headings across adjacent paragraphs.
    6) Normalize headings and handle inline heading text.
    7) Insert Abstract/Introduction if missing with heuristics.
    8) Normalize other headings again and return joined Markdown text.
    
    metadata_action: "extract" | "keep" | "remove"
    """
    if metadata_action not in {"extract", "keep", "remove"}:
        raise ValueError("metadata_action must be 'extract', 'keep' or 'remove'")

    text = normalize_ocr(text)
    paragraphs = split_paragraphs(text)
    
    # Remove duplicate title lines (NEW)
    paragraphs = remove_duplicate_titles(paragraphs)

    # Extract metadata
    metadata_text = None
    if metadata_action != "keep":
        paragraphs, metadata_text = extract_metadata(paragraphs)

    # Join broken multi-line headings
    paragraphs = join_broken_headings(paragraphs)

    # Normalize obvious headings first
    normalized = []
    for p in paragraphs:
        # detect "Abstract: rest" or "A B S T R A C T: rest"
        m = re.match(r"^\s*(?:A\s*B\s*S\s*T\s*R\s*A\s*C\s*T|abstract)\s*:\s*(.+)$", p, flags=re.I)
        if m:
            rest = m.group(1).strip()
            normalized.append("## Abstract")
            if rest:
                normalized.append(rest)
            continue
        normalized.append(p)
    paragraphs = normalized

    # Normalize other headings
    paragraphs = normalize_other_headings(paragraphs)

    # Insert abstract & intro with improved heuristics
    paragraphs = insert_abstract_and_intro_improved(paragraphs)

    # Normalize headings again
    paragraphs = normalize_other_headings(paragraphs)

    # Build final output
    final_paragraphs = []
    if metadata_action == "extract" and metadata_text:
        final_paragraphs.append("## Metadata\n\n" + metadata_text)
        final_paragraphs.append("")

    final_paragraphs.extend(paragraphs)

    # Remove duplicate consecutive headings
    cleaned = []
    prev = None
    for p in final_paragraphs:
        p_stripped = p.strip()
        if prev and prev.lower().startswith("##") and p_stripped.lower().startswith("##") and prev.lower() == p_stripped.lower():
            continue
        cleaned.append(p_stripped)
        prev = p_stripped

    return join_paragraphs(cleaned).strip() + "\n"


# -------------------------
# Demo / quick test
# -------------------------
if __name__ == "__main__":
    # Example usage
    import sys
    
    # You can replace this with your file path
    test_file = """H://python_projects//scientific//olmOCR//pdfs//raw ocr//Host–Guest Interactions in a Metal–Organic Framework Isoreticular Series for Molecular Photocatalytic CO2 Reduction._attempt 2_.md//_cleaned_Host–Guest Interactions in a Metal–Organic Framework Isoreticular Series for Molecular Photocatalytic CO2 Reduction_extracted.md"""
    
    try:
        with open(test_file, "r", encoding="utf8") as f:
            ocr = f.read()
            fixed = process_paper(ocr, metadata_action="keep")
        
        with open("fixed_heading.md", "w", encoding="utf8") as f:
            f.write(fixed)
        print("Fixed text written to fixed_heading.md")
        
    except FileNotFoundError:
        print(f"File not found: {test_file}")
        print("\nUsing example text instead...")
        
        # Example text with the issues you mentioned
        example_text = """
## Abstract

Molybdenum Sulfide within a Metal-Organic Framework for Photocatalytic Hydrogen Evolution from Water

## Introduction

Molybdenum Sulfide within a Metal-Organic Framework for Photocatalytic Hydrogen Evolution from Water

Molybdenum Sulfide within a Metal-Organic Framework for Photocatalytic Hydrogen Evolution from Water

Hyunho Noh, Ying Yang, Sol Ahn, Aaron W. Peters, Omar K. Farha, and Joseph T. Hupp

1Department of Chemistry, Northwestern University, Evanston, Illinois 60208, USA
2Department of Chemical and Biological Engineering, Northwestern University, Evanston, Illinois 60208, USA

A representative metal-organic framework, NU-1000, was functionalized with MoSx. The previously determined crystal structure of the material, named MoSx-SIM, consists of monometallic Mo(IV) ions with two sulfhydryl ligands. The metal ions are anchored to the framework by displacing protons presented by the -OH/-OH2 groups on the Zr6 node. As shown previously, the MOF-supported complexes are electrocatalytic for hydrogen evolution from acidified water. The earlier electrocatalysis results, together with the nearly ideal formal potential of the Mo(IV)/II couple (i.e., nearly coincident with that of the hydrogen couple), and the physical proximity of UV-absorbing MOF linkers to the complexes, suggested to us that the linkers might behave photosensitizers for catalyst reduction, and subsequently, for H2 evolution from water. To our surprise, MoSx-SIM, when UV-illuminated in an aqueous buffer at near-neutral pH, displays a biphasic photocatalytic response: an initially slow rate of reaction, i.e. 0.56 mmol g−1 h−1, followed by an increase to 4 mmol g−1 h−1. Ex-situ catalyst examination revealed that nanoparticulate MoSx suspended within the reaction mixture is the actual catalyst. Thus, photo-assisted restructuring and detachment of the catalyst or pre-catalyst from the MOF node appears to be necessary for the catalyst to reduce water at neutral pH.

© The Author(s) 2019. Published by ECS. This is an open access article distributed under the terms of the Creative Commons Attribution 4.0 License (CC BY, http://creativecommons.org/licenses/by/4.0/), which permits unrestricted reuse of the work in any medium, provided the original work is properly cited. [DOI: 10.1149/2.0261905jes]

Manuscript submitted November 12, 2018; revised manuscript received January 11, 2019. Published January 24, 2019. This paper is part of the JES Focus Issue on Semiconductor Electrochemistry and Photoelectrochemistry in Honor of Krishnan Rajeshwar.

Molecular hydrogen is a compound of undeniable global importance given its high demand as a feedstock in the industrial scale production of hydrocarbons and ammonia (i.e. Fisher-Tropsch and Haber-Bosch processes, respectively)1-5 and as a reactant in fuel cells,6,7 a technology of existing and increasing commercial significance. The overwhelming majority of commercially used H2 is obtained by methane steam reforming.
        """
        
        fixed = process_paper(example_text, metadata_action="keep")
        print("\nFixed text:\n")
        print(fixed)
# -------------------------
# Demo / quick test
# -------------------------

if __name__ == "__main__":
    with open("""H://python_projects//scientific//olmOCR//pdfs//raw ocr//Host–Guest Interactions in a Metal–Organic Framework Isoreticular Series for Molecular Photocatalytic CO2 Reduction._attempt 2_.md//_cleaned_Host–Guest Interactions in a Metal–Organic Framework Isoreticular Series for Molecular Photocatalytic CO2 Reduction_extracted.md""","r",encoding="utf8") as f:
        ocr = f.read()
        fixed= process_paper(ocr,metadata_action="keep")

    with open("fixed_heading.md","w",encoding="utf8") as f:
        f.write(fixed)
    print("fixed text:\n")