import difflib
import re

# -----------------------------
# 1. Normalization map
# -----------------------------
NORMALIZATION_MAP = {
    # Units
    r"\bdegC\b": "°C", r"\boC\b": "°C", r"\bCelsius\b": "°C",
    r"\bK\b": "K", r"\bdeg\b": "°",

    # Microns / micro
    r"\bum\b": "µm", r"\\mu\s?m": "µm",

    # Spectroscopy / XRD
    r"cm\^-1": "cm⁻¹", r"\\AA": "Å", r"\\angstrom": "Å", r"\b2theta\b": "2θ",

    # Fractions
    r"\\frac\{1\}\{2\}": "½", r"\\frac\{1\}\{3\}": "⅓", r"\\frac\{2\}\{3\}": "⅔",
    r"\\frac\{1\}\{4\}": "¼", r"\\frac\{3\}\{4\}": "¾",

    # Superscripts / subscripts digits
    r"\^2\b": "²", r"\^3\b": "³", r"\^-\s?1\b": "⁻¹",
    r"_0\b": "₀", r"_1\b": "₁", r"_2\b": "₂", r"_3\b": "₃", r"_4\b": "₄", r"_5\b": "₅",
    r"_6\b": "₆", r"_7\b": "₇", r"_8\b": "₈", r"_9\b": "₉",

    # Greek letters
    r"\\alpha": "α", r"\\beta": "β", r"\\gamma": "γ", r"\\delta": "δ",
    r"\\theta": "θ", r"\\lambda": "λ", r"\\mu": "µ", r"\\sigma": "σ", r"\\omega": "ω",

    # Math / Physics symbols
    r"\\sim": "∼", r"~": "∼", r"\\pm": "±", r"\+/-": "±", r"\+-": "±",
    r"\\times": "×", r"\\rightarrow": "→", r"\\leftarrow": "←",
    r"\\Rightarrow": "⇒", r"\\Leftarrow": "⇐", r"\\leq": "≤", r"\\geq": "≥",

    # Ligatures / OCR quirks
    r"ﬁ": "fi", r"ﬂ": "fl",

    # Crystallography brackets
    r"\[([0-9]+)\]": r"[\1]", r"\{([0-9]+)\}": r"{\1}", r"⟨([0-9]+)⟩": r"⟨\1⟩",

    # Chemical formulas common subscripts
    r"O2\b": "O₂", r"CO2\b": "CO₂", r"H2O\b": "H₂O", r"TiO2\b": "TiO₂",
}

# Heuristic for OCR degree symbol
_DEGREE_O_PATTERN = re.compile(r"(?<=\d)\s*[oO]\s*(?=(?:[CFK]|°|\b))")

def normalize_text(text: str) -> str:
    """Apply aggressive normalization for materials science OCR text."""
    if not text:
        return ""
    # LaTeX & OCR replacements
    for pattern, repl in NORMALIZATION_MAP.items():
        text = re.sub(pattern, repl, text, flags=re.IGNORECASE)
    # Optional degree heuristic
    text = _DEGREE_O_PATTERN.sub("°", text)
    # Remove stray braces
    text = re.sub(r"[{}]", "", text)
    # Normalize ligatures / spaces
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\s*\n\s*", "\n", text)
    return text.strip()

# -----------------------------
# 2. Merge OCR outputs
# -----------------------------
def merge_ocr_texts(text1: str, text2: str) -> str:
    """
    Merge two OCR versions:
    - Aligns tokens using difflib
    - Keeps missing content from either version
    - Normalizes aggressively for LLM use
    """
    # Normalize each text first
    norm1 = normalize_text(text1)
    norm2 = normalize_text(text2)

    tokens1 = norm1.split()
    tokens2 = norm2.split()
    s = difflib.SequenceMatcher(None, tokens1, tokens2)
    merged_tokens = []

    for tag, i1, i2, j1, j2 in s.get_opcodes():
        if tag in ("equal", "replace"):
            # Take the version with more content if replaced
            merged_tokens.extend(tokens1[i1:i2] if len(tokens1[i1:i2]) >= len(tokens2[j1:j2]) else tokens2[j1:j2])
        elif tag == "delete":
            merged_tokens.extend(tokens1[i1:i2])
        elif tag == "insert":
            merged_tokens.extend(tokens2[j1:j2])

    return " ".join(merged_tokens)

# -----------------------------
# 3. Example usage
# -----------------------------
if __name__ == "__main__":
    ocr1 = "The sample was heated to 90 oC. The size was ~10 um. TiO2 formed."
    ocr2 = "The sample was heated to 90 degC. The size was \\sim 10 \\mu m. TiO₂ formed."

    merged = merge_ocr_texts(ocr1, ocr2)
    print("Merged and normalized text:\n")
    print(merged)
