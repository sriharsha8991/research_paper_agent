"""LaTeX Generator — renders GeneratedPaper into compilable LaTeX source.

Implements §10 of plan_doc_v2.md:
- NeurIPS/ICLR/ICML style templates
- Section-by-section rendering
- Table/figure integration
- BibTeX handling
- PDF compilation via pdflatex
"""

from __future__ import annotations

import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from article_generator.config import AppConfig
from article_generator.models import GeneratedPaper, VenueStyle

logger = logging.getLogger(__name__)


# Minimal LaTeX templates per venue style
TEMPLATES = {
    VenueStyle.NEURIPS: r"""\documentclass{article}
\usepackage[utf8]{inputenc}
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{graphicx}
\usepackage{hyperref}
\usepackage{natbib}
\usepackage[margin=1in]{geometry}

\title{%(title)s}
\author{Research Paper Generator}
\date{}

\begin{document}
\maketitle

%(abstract)s

%(body)s

\bibliographystyle{plainnat}
%(bibliography)s

\end{document}
""",
}


SECTION_ORDER = [
    ("abstract", "Abstract"),
    ("introduction", "Introduction"),
    ("literature_review", "Related Work"),
    ("method_math", "Method: Mathematical Framework"),
    ("method_arch", "Method: Architecture"),
    ("experiments", "Experiments"),
    ("discussion", "Discussion"),
    ("conclusion", "Conclusion"),
]


class LaTeXGenerator:
    """Generate LaTeX source from a GeneratedPaper."""

    def __init__(self, config: AppConfig):
        self.config = config
        self.output_dir = Path(config.paths.data_dir) / "output"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate(self, paper: GeneratedPaper, venue: VenueStyle = VenueStyle.NEURIPS) -> str:
        """Generate complete LaTeX source."""
        template = TEMPLATES.get(venue, TEMPLATES[VenueStyle.NEURIPS])

        # Build abstract
        abstract_text = paper.sections.get("abstract", "")
        abstract_latex = ""
        if abstract_text:
            abstract_latex = (
                "\\begin{abstract}\n"
                f"{self._escape_latex(abstract_text)}\n"
                "\\end{abstract}"
            )

        # Build body sections
        body_parts = []
        for key, heading in SECTION_ORDER:
            if key == "abstract":
                continue  # Handled separately
            content = paper.sections.get(key, "")
            if content:
                body_parts.append(
                    f"\\section{{{heading}}}\n"
                    f"{self._escape_latex(content)}"
                )

        body = "\n\n".join(body_parts)

        # Bibliography
        if paper.bibtex:
            bib_section = "\\bibliography{references}"
        else:
            bib_section = ""

        latex_source = template % {
            "title": self._escape_latex(paper.title),
            "abstract": abstract_latex,
            "body": body,
            "bibliography": bib_section,
        }

        paper.latex_source = latex_source
        return latex_source

    def save(self, paper: GeneratedPaper, filename: str = "paper") -> Path:
        """Save LaTeX source and BibTeX to output directory."""
        if not paper.latex_source:
            self.generate(paper)

        tex_path = self.output_dir / f"{filename}.tex"
        tex_path.write_text(paper.latex_source, encoding="utf-8")
        logger.info("LaTeX source saved to %s", tex_path)

        if paper.bibtex:
            bib_path = self.output_dir / "references.bib"
            bib_path.write_text(paper.bibtex, encoding="utf-8")
            logger.info("BibTeX saved to %s", bib_path)

        # Save confidence report as supplementary
        report_path = self.output_dir / f"{filename}_report.txt"
        self._save_report(paper, report_path)

        return tex_path

    def compile_pdf(self, tex_path: Path) -> Path | None:
        """Compile LaTeX to PDF using pdflatex."""
        try:
            for _ in range(2):  # Run twice for references
                result = subprocess.run(
                    ["pdflatex", "-interaction=nonstopmode", str(tex_path)],
                    cwd=str(tex_path.parent),
                    capture_output=True,
                    text=True,
                    timeout=60,
                )

            # Check if bibtex is needed
            bib_path = tex_path.parent / "references.bib"
            if bib_path.exists():
                aux_path = tex_path.with_suffix(".aux")
                subprocess.run(
                    ["bibtex", str(aux_path.stem)],
                    cwd=str(tex_path.parent),
                    capture_output=True,
                    timeout=30,
                )
                # Recompile twice more
                for _ in range(2):
                    subprocess.run(
                        ["pdflatex", "-interaction=nonstopmode", str(tex_path)],
                        cwd=str(tex_path.parent),
                        capture_output=True,
                        timeout=60,
                    )

            pdf_path = tex_path.with_suffix(".pdf")
            if pdf_path.exists():
                logger.info("PDF compiled: %s", pdf_path)
                return pdf_path
            else:
                logger.warning("PDF compilation produced no output")
                return None

        except FileNotFoundError:
            logger.info("pdflatex not found — LaTeX source saved but PDF not compiled")
            return None
        except subprocess.TimeoutExpired:
            logger.warning("PDF compilation timed out")
            return None

    def _escape_latex(self, text: str) -> str:
        """Escape special LaTeX characters (but preserve LaTeX commands)."""
        # Don't escape if text contains LaTeX commands
        if "\\" in text and any(cmd in text for cmd in ["\\begin", "\\end", "\\cite", "\\ref", "\\textbf"]):
            return text

        replacements = {
            "&": r"\&",
            "%": r"\%",
            "#": r"\#",
            "_": r"\_",
        }
        for char, escaped in replacements.items():
            # Don't escape if already escaped
            text = text.replace(f"\\{char}", f"__ESCAPED_{char}__")
            text = text.replace(char, escaped)
            text = text.replace(f"__ESCAPED_{char}__", f"\\{char}")

        return text

    def _save_report(self, paper: GeneratedPaper, path: Path) -> None:
        """Save a human-readable quality report."""
        lines = [
            f"Quality Report for: {paper.title}",
            "=" * 60,
            "",
        ]

        if paper.confidence_report and hasattr(paper.confidence_report, "overall_confidence"):
            cr = paper.confidence_report
            lines.extend([
                f"Overall Confidence: {cr.overall_confidence:.2%}",
                f"Hallucination Risk: {cr.hallucination_risk}",
                f"Recommendation: {cr.recommendation}",
                "",
            ])

        if paper.novelty_report and hasattr(paper.novelty_report, "overall"):
            nr = paper.novelty_report
            lines.extend([
                "Novelty Scores:",
                f"  Textual: {nr.textual_novelty:.3f}",
                f"  Structural: {nr.structural_novelty:.3f}",
                f"  Contribution: {nr.contribution_novelty:.3f}",
                f"  Overall: {nr.overall:.3f}",
                "",
            ])

        if paper.generation_log:
            lines.extend([
                "Generation Log:",
                f"  Stages completed: {paper.generation_log.get('stages_completed', [])}",
                f"  Stages failed: {paper.generation_log.get('stages_failed', [])}",
            ])

        path.write_text("\n".join(lines), encoding="utf-8")
