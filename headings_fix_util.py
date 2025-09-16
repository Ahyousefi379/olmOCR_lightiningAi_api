import re
from typing import List, Dict

class ScientificTextToMarkdownConverter:
    """
    Converts OCR'd scientific paper text into a structured Markdown file.

    The process is as follows:
    1.  Define a comprehensive list of possible section titles and their variations.
    2.  Find explicitly mentioned section titles in the text, handling OCR noise.
    3.  If key sections (References, Abstract, Introduction) are not found,
        use robust heuristics and pattern matching to locate them implicitly.
    4.  Resolve any overlaps or conflicts between found sections.
    5.  Assemble the final Markdown text with Level 1 headings for identified sections.
    """

    SECTION_TITLES = {
        "Abstract": ["abstract", "summary"],
        "Introduction": ["introduction"],
        "Materials and Methods": ["materials and methods", "experimental section",
                                  "experimental details", "experimental", "methods",
                                  "methodology"],
        "Results": ["results"],
        "Discussion": ["discussion"],
        "Results and Discussion": ["results and discussion", r"results & discussion"],
        "Conclusion": ["conclusion", "conclusions", "concluding remarks"],
        "Acknowledgements": ["acknowledgements", "acknowledgment", "funding"],
        "Author Contributions": ["author contributions", "author contribution"],
        "Competing Interests": ["competing interests", "declaration of competing interest"],
        "Supporting Information": ["supporting information", "supplementary information",
                                   "supplementary data"],
        "References": ["references", "bibliography", "literature cited"],
    }

    def __init__(self, text: str):
        # Remove markdown headers from input if they exist
        self.raw_text = re.sub(r'^\s*#\s+', '', text, flags=re.MULTILINE)
        self.lines = self.raw_text.split('\n')
        self.sections: List[Dict] = []

    def _normalize_line_for_title_check(self, line: str) -> str:
        processed = re.sub(r'^\s*\d*[\.KATEX_INLINE_CLOSE]?\s*', '', line.lower())
        processed = re.sub(r'[^a-z\s&]', '', processed)
        processed = re.sub(r'\s+', '', processed)
        return processed

    def _find_explicit_sections(self):
        found_titles = []
        for i, line in enumerate(self.lines):
            if not line.strip() or len(line.strip()) > 70:
                continue
            
            # Handle titles that start with "ABSTRACT:" etc.
            line_to_check = line.split(':')[0] if ':' in line else line
            normalized_line = self._normalize_line_for_title_check(line_to_check)
            if not normalized_line:
                continue

            for canonical_name, variations in self.SECTION_TITLES.items():
                for variation in variations:
                    normalized_variation = re.sub(r'\s+', '', variation)
                    if normalized_line == normalized_variation:
                        if not any(s['start'] == i for s in found_titles):
                            found_titles.append({'name': canonical_name, 'start': i, 'source': 'explicit'})
                        break
        
        found_titles.sort(key=lambda x: x['start'])
        for section in found_titles:
            self.sections.append(section)
        
        print(f"Found {len(self.sections)} explicit sections.")

    def _has_section(self, name: str) -> bool:
        return any(s['name'] == name for s in self.sections)

    def _is_reference_like(self, line: str) -> bool:
        line = line.strip()
        if not line: return False
        ref_pattern = re.compile(
            r'^\s*(```math\d+```|\d+\.|KATEX_INLINE_OPEN\d+KATEX_INLINE_CLOSE)' r'|' r'(```math\d+```)' r'|'
            r'(\b(19|20)\d{2}\b)' r'|' r'(et al\.)' r'|' r'(doi:)'
        )
        return bool(ref_pattern.search(line))

    def _find_implicit_references(self):
        if self._has_section("References"): return

        consecutive_ref_count = 0
        potential_start = -1
        
        for i in range(len(self.lines) - 1, 0, -1):
            if self._is_reference_like(self.lines[i]):
                consecutive_ref_count += 1
                potential_start = i
            elif self.lines[i].strip() == "":
                continue
            else:
                if consecutive_ref_count >= 3: break
                else:
                    consecutive_ref_count = 0
                    potential_start = -1
        
        if consecutive_ref_count >= 3 and potential_start != -1:
            print("Found implicit 'References' section.")
            self.sections.append({'name': "References", 'start': potential_start, 'source': 'implicit'})

    def _find_implicit_abstract(self):
        if self._has_section("Abstract"): return

        paragraphs = self.raw_text.split('\n\n')
        ref_citation_pattern = re.compile(r'```math\d{1,3}```')

        for i, p in enumerate(paragraphs[:5]):
            p_clean = p.strip()
            if 300 < len(p_clean) < 3000 and not ref_citation_pattern.search(p_clean) and "doi:" not in p_clean.lower():
                try:
                    start_line = self.raw_text[:self.raw_text.index(p_clean)].count('\n')
                    if not any(s['start'] == start_line for s in self.sections):
                        print("Found implicit 'Abstract' section.")
                        self.sections.append({'name': "Abstract", 'start': start_line, 'source': 'implicit'})
                        return
                except ValueError:
                    continue

    def _find_implicit_introduction(self):
        """
        Robustly finds the introduction by anchoring to the end of the abstract block
        and intelligently skipping intermediate metadata like 'KEYWORDS'.
        """
        if self._has_section("Introduction"): return

        # Anchor finding the Introduction to the Abstract, which is more reliable
        abstract_section = next((s for s in self.sections if s['name'] == "Abstract"), None)
        
        # Start search from line 0 if no abstract is found, otherwise start after it
        search_start_line = 0
        if abstract_section:
            # Find the end of the abstract paragraph (the first blank line after it starts)
            abstract_end_line = -1
            for i in range(abstract_section['start'], len(self.lines)):
                if not self.lines[i].strip():
                    abstract_end_line = i
                    break
            search_start_line = abstract_end_line if abstract_end_line != -1 else abstract_section['start'] + 1

        # Now, find the first *real* content line, skipping metadata
        intro_start_line = -1
        for i in range(search_start_line, len(self.lines)):
            line = self.lines[i].strip()
            if not line:
                continue # Skip empty lines

            # Heuristic to skip metadata lines like "KEYWORDS: ..."
            if ":" in line and len(line) < 150 and not line.endswith('.'):
                continue
            
            # Found the first line of real content
            intro_start_line = i
            break
        
        if intro_start_line == -1: return # No content found

        # Ensure we don't accidentally label an existing section as the introduction
        if any(s['start'] == intro_start_line for s in self.sections):
            return

        print("Found implicit 'Introduction' section.")
        self.sections.append({'name': "Introduction", 'start': intro_start_line, 'source': 'implicit'})

    def _resolve_section_boundaries(self):
        if not self.sections: return
        
        self.sections.sort(key=lambda x: x['start'])
        
        unique_sections = []
        seen_starts = set()
        for section in self.sections:
            if section['start'] not in seen_starts:
                unique_sections.append(section)
                seen_starts.add(section['start'])
            else:
                print(f"Resolved duplicate section at line {section['start']}, keeping first one found.")
        self.sections = unique_sections

        for i, section in enumerate(self.sections):
            next_section_start = self.sections[i+1]['start'] if i + 1 < len(self.sections) else len(self.lines)
            section['end'] = next_section_start
        
        print("\n--- Resolved Section Boundaries ---")

    def assemble_markdown(self) -> str:
        if not self.sections:
            print("Warning: No sections were identified. Returning original text.")
            return self.raw_text

        markdown_parts = []
        
        first_section_start = self.sections[0]['start']
        if first_section_start > 0:
            preamble = "\n".join(self.lines[0:first_section_start]).strip()
            if preamble:
                markdown_parts.append(preamble)
        
        for section in self.sections:
            markdown_parts.append(f"\n# {section['name']}\n")
            
            content_lines = self.lines[section['start']:section.get('end', len(self.lines))]
            
            if section['source'] == 'explicit' and content_lines:
                first_line = content_lines[0]
                normalized_first_line = self._normalize_line_for_title_check(first_line.split(':')[0])
                normalized_section_name = self._normalize_line_for_title_check(section['name'])
                if normalized_first_line in normalized_section_name or normalized_section_name in normalized_first_line:
                    # Special case for "ABSTRACT: ..." where content is on the same line
                    if ':' in first_line:
                        content_lines[0] = first_line.split(':', 1)[1].strip()
                    else:
                        content_lines = content_lines[1:]

            section_content = "\n".join(content_lines).strip()
            markdown_parts.append(section_content)
        
        return "\n\n".join(markdown_parts)

    def convert(self) -> str:
        print("--- Starting Conversion Process ---")
        self._find_explicit_sections()
        self._find_implicit_references()
        self._find_implicit_abstract()
        self._find_implicit_introduction()
        self._resolve_section_boundaries()
        
        print("\n--- Final Sections Identified ---")
        if self.sections:
            # Sort one last time before printing
            self.sections.sort(key=lambda x: x['start'])
            for s in self.sections:
                print(f"- {s['name']} (Lines {s.get('start', 'N/A')}-{s.get('end', 'N/A')}) [Source: {s['source']}]")
        else:
            print("None.")
        
        print("\n--- Assembling Markdown ---")
        markdown = self.assemble_markdown()
        print("--- Conversion Complete ---")
        return markdown

# --- EXAMPLE USAGE WITH YOUR NEW TEXT ---

your_new_sample_text = """
miRNA-Guided Imaging and Photodynamic Therapy Treatment of Cancer Cells Using Zn(II)-Protoporphyrin IX-Loaded Metal–Organic Framework Nanoparticles

Pu Zhang, Yu Ouyang, Yang Sung Sohn, Michael Fadeev, Ola Karmi, Rachel Nechushtai, Ilan Stein, Eli Pikarsky, and Itamar Willner*

ABSTRACT: An analytical platform for the selective miRNA-21-guided imaging of breast cancer cells and miRNA-221-guided imaging of ovarian cancer cells and the selective photodynamic therapy (PDT) of these cancer cells is introduced. The method is based on Zn(II)-protoporphyrin IX, Zn(II)-PPIX-loaded UiO-66 metal–organic framework nanoparticles, NMOFs, gated by two hairpins H$_h$/H$_p$ through ligation of their phosphate residues to the vacant Zn$^{2+}$-ions associated with the NMOFs. The hairpins are engineered to include the miRNA recognition sequence in the stem domain of H$_h$, and in the H$_h$ and H$_p$, partial locked stem regions of G-quadruplex subunits. Intracellular phosphate-ions displace the hairpins, resulting in the release of the Zn(II)-PPIX and intracellular miRNAs open H$_h$, and this triggers the autonomous cross-opening of H$_h$ and H$_p$. This activates the interhairpin hybridization chain reaction and leads to the assembly of highly fluorescent Zn(II)-PPIX-loaded G-quadruplex chains. The miRNA-guided fluorescent chains allow selective imaging of cancer cells. Moreover, PDT with visible light selectively kills cancer cells and tumor cells through the formation of toxic reactive oxygen species.

KEYWORDS: fluorescence, G-quadruplexes, hybridization chain reaction, breast cancer, ovarian cancer, reactive oxygen species

MicroRNAs, miRNAs, are short noncoding RNA sequences that regulate gene expression in multiple cellular adaptations. Up-regulation or down-regulation of miRNAs has been related to numerous biological processes, for example, cell proliferation, cell aging, apoptosis, and different diseases. Particularly, the identification of miRNAs related to different malignant cells finds growing interest, and miRNAs are important biomarkers for diagnosis, prognosis, progression, and recurrence of cancer. Different analytical methods to detect miRNAs were developed, including electrochemical, optical, and nanoparticle-based platforms. The low levels of intracellular miRNAs required, however, the development of amplified miRNA detection platforms. Different in vitro amplified miRNA detection schemes were reported, for example, polymerase-mediated rolling circle amplification, exponential exonuclease- or nuclease-assisted amplified detection of miRNAs, and DNAzyme-catalyzed analysis of miRNAs. Also, enzyme-free amplified sensing platforms of miRNAs were demonstrated, including the hybridization chain reaction (HCR), catalytic DNA hairpin assembly, and photoactivated toehold-mediated strand displacement process. In addition, the multiplexed analyses of miRNAs, multifunctional DNA nanostructures, and miRNA arrays, for parallel and high-throughput detection of the biomarkers, were reported.

The intracellular detection of miRNAs is particularly important for miRNAs imaging in cancer cells. Graphene oxide, polydopamine/ZnO nanoparticles, and carbon nitride were used as carriers for nucleic acid activating...
the intracellular HCR visualizing miRNAs. Similarly, an elegant miRNA-triggered concatenated HCR using functional DNA hairpins\\textsuperscript{33} and fluorophore-labeled DNA hairpins-loaded Au nanoparticles for imaging intracellular miRNAs\\textsuperscript{34,35} was demonstrated. miRNAs-triggered release of hairpins and catalytic hairpin assembly regenerating the miRNAs provided a useful method for the amplified imaging of miRNA-containing cells.\\textsuperscript{36,37} Also, the collective intracellular imaging of miRNAs, the spatiotemporal fluorescence and electrochemical detection of miRNA, at single cell level was demonstrated.\\textsuperscript{38}
"""

converter = ScientificTextToMarkdownConverter(your_new_sample_text)
markdown_output = converter.convert()

print("\n\n--- FINAL MARKDOWN OUTPUT ---\n")
print(markdown_output)