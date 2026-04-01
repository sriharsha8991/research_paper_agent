"""Main entry point for the Article Generator CLI.

Usage:
    article-gen --title "Your Paper Title" [options]
    python -m article_generator.main --title "..." --abstract "..." --venue neurips
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from article_generator.config import AppConfig
from article_generator.models import GenerationInput, VenueStyle
from article_generator.orchestrator.pipeline import PipelineOrchestrator
from article_generator.output.latex_gen import LaTeXGenerator


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    # Quiet noisy libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="article-gen",
        description="AI/ML Research Paper Generator — enterprise-grade multi-agent pipeline",
    )
    parser.add_argument(
        "--title", required=True,
        help="Paper title (required)",
    )
    parser.add_argument(
        "--abstract", default=None,
        help="Draft abstract (optional — will be generated if not provided)",
    )
    parser.add_argument(
        "--direction", default=None,
        help="Research direction / key ideas to explore",
    )
    parser.add_argument(
        "--venue", default="neurips",
        choices=[v.value for v in VenueStyle],
        help="Target venue style (default: neurips)",
    )
    parser.add_argument(
        "--max-pages", type=int, default=15,
        help="Maximum paper length in pages (default: 15)",
    )
    parser.add_argument(
        "--must-cite", nargs="*", default=[],
        help="Paper IDs that must be cited",
    )
    parser.add_argument(
        "--config", default=None,
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output directory (default: data/output)",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable debug logging",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Validate config and print plan without executing",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)

    logger = logging.getLogger("article_generator")
    logger.info("Article Generator starting...")

    # Load config
    config = AppConfig()
    if args.config:
        config = AppConfig.from_yaml(args.config)

    if args.output:
        config.paths.data_dir = Path(args.output).parent
        Path(args.output).mkdir(parents=True, exist_ok=True)

    # Build generation input
    gen_input = GenerationInput(
        title=args.title,
        abstract=args.abstract,
        research_direction=args.direction,
        target_venue=VenueStyle(args.venue),
    )
    gen_input.constraints.max_pages = args.max_pages
    gen_input.constraints.must_cite = args.must_cite

    if args.dry_run:
        logger.info("Dry run — configuration validated")
        logger.info("Title: %s", gen_input.title)
        logger.info("Venue: %s", gen_input.target_venue.value)
        logger.info("Config: %s", config.model_dump_json(indent=2))
        return

    # Run pipeline
    orchestrator = PipelineOrchestrator(config)
    paper = orchestrator.run(gen_input)

    # Generate LaTeX
    latex_gen = LaTeXGenerator(config)
    latex_source = latex_gen.generate(paper, gen_input.target_venue)
    tex_path = latex_gen.save(paper)

    # Try to compile PDF
    pdf_path = latex_gen.compile_pdf(tex_path)

    # Summary
    print("\n" + "=" * 60)
    print(f"Paper generated: {paper.title}")
    print(f"LaTeX source: {tex_path}")
    if pdf_path:
        print(f"PDF: {pdf_path}")
    print(f"Sections: {list(paper.sections.keys())}")
    if hasattr(paper.confidence_report, "overall_confidence"):
        print(f"Confidence: {paper.confidence_report.overall_confidence:.2%}")
        print(f"Hallucination risk: {paper.confidence_report.hallucination_risk}")
    if hasattr(paper.novelty_report, "overall"):
        print(f"Novelty: {paper.novelty_report.overall:.3f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
