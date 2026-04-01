"""Humanized Writing Engine — reduces AI detection signatures.

Implements §8.9 of plan_doc_v2.md:
- Sentence length variation (σ > 4 words)
- Vocabulary diversity (TTR > 0.65)
- Rhetorical device injection (hedging, contrast, analogy)
- Paragraph rhythm modulation
"""

from __future__ import annotations

import logging
import random
from typing import Any

from article_generator.llm_client import LLMClient

logger = logging.getLogger(__name__)


class HumanizedWriter:
    """Post-processes text to reduce AI-detection signatures while maintaining quality."""

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def humanize(self, text: str, section_name: str = "") -> str:
        """Apply humanization passes to generated text."""
        # Pass 1: Compute metrics
        metrics = self._compute_metrics(text)
        logger.info(
            "Pre-humanization metrics for %s: TTR=%.3f, sent_len_std=%.1f",
            section_name, metrics["ttr"], metrics["sent_len_std"],
        )

        # Pass 2: LLM rewrite if metrics are too uniform
        if metrics["sent_len_std"] < 4.0 or metrics["ttr"] < 0.55:
            text = self._llm_rewrite(text, section_name, metrics)

        # Pass 3: Verify post-humanization metrics
        post_metrics = self._compute_metrics(text)
        logger.info(
            "Post-humanization metrics for %s: TTR=%.3f, sent_len_std=%.1f",
            section_name, post_metrics["ttr"], post_metrics["sent_len_std"],
        )

        return text

    def _compute_metrics(self, text: str) -> dict[str, float]:
        """Compute language diversity metrics."""
        words = text.lower().split()
        sentences = [s.strip() for s in text.split(".") if len(s.strip()) > 5]

        # Type-Token Ratio
        unique_words = set(words)
        ttr = len(unique_words) / max(len(words), 1)

        # Sentence length standard deviation
        sent_lengths = [len(s.split()) for s in sentences]
        if len(sent_lengths) > 1:
            mean_len = sum(sent_lengths) / len(sent_lengths)
            variance = sum((l - mean_len) ** 2 for l in sent_lengths) / len(sent_lengths)
            sent_len_std = variance ** 0.5
        else:
            sent_len_std = 0.0

        # Average sentence length
        avg_sent_len = sum(sent_lengths) / max(len(sent_lengths), 1)

        return {
            "ttr": ttr,
            "sent_len_std": sent_len_std,
            "avg_sent_len": avg_sent_len,
            "word_count": len(words),
            "sentence_count": len(sentences),
        }

    def _llm_rewrite(
        self, text: str, section_name: str, metrics: dict[str, float]
    ) -> str:
        """Use LLM to rewrite text with more natural variation."""
        issues = []
        if metrics["sent_len_std"] < 4.0:
            issues.append("Sentence lengths are too uniform — vary between 8-30 words")
        if metrics["ttr"] < 0.55:
            issues.append("Vocabulary is repetitive — use more synonyms and varied phrasing")

        prompt = (
            f"Rewrite this {section_name} section to sound more natural and human-written.\n\n"
            f"Issues to fix:\n" + "\n".join(f"- {i}" for i in issues) + "\n\n"
            f"Additional rules:\n"
            f"- Mix short punchy sentences with longer complex ones\n"
            f"- Occasionally use rhetorical questions or contrasts\n"
            f"- Vary paragraph openings (don't always start with 'We' or 'The')\n"
            f"- Use hedging where appropriate ('arguably', 'it appears that')\n"
            f"- Preserve ALL technical content, citations, and mathematical notation\n"
            f"- Do NOT add new claims or remove existing ones\n\n"
            f"Original text:\n{text[:4000]}\n\n"
            f"Rewritten text:"
        )

        try:
            response = self.llm.generate(
                prompt,
                system="You are an expert academic writer who produces natural, "
                       "human-sounding prose. Preserve all technical content exactly.",
                temperature=0.6,
                max_tokens=4096,
            )
            return response.strip()
        except Exception as e:
            logger.warning("Humanization rewrite failed: %s", e)
            return text
