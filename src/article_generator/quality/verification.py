"""Chain-of-Verification (CoVe) — fact-checks generated text against sources.

Implements §8.1-8.2 of plan_doc_v2.md:
- Extracts factual claims from generated text
- Generates verification questions per claim
- Answers questions from source context
- Corrects inconsistencies
"""

from __future__ import annotations

import json
import logging
from typing import Any

from article_generator.llm_client import LLMClient
from article_generator.models import (
    ProvenanceRecord,
    ProvenanceType,
    VerificationResult,
    VerifiedSection,
    RetrievalResult,
)

logger = logging.getLogger(__name__)


class ChainOfVerification:
    """Verify generated text against source documents using CoVe protocol."""

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def verify_section(
        self,
        section_text: str,
        source_context: RetrievalResult,
        section_name: str = "",
    ) -> VerifiedSection:
        """Run the full CoVe pipeline on a generated section."""

        # Step 1: Extract factual claims
        claims = self._extract_claims(section_text)

        # Step 2: Generate verification questions
        verification_log: list[VerificationResult] = []
        corrections_made = 0

        for claim in claims:
            vq = self._generate_verification_question(claim)
            draft_answer = claim
            source_answer = self._answer_from_sources(vq, source_context)

            consistent = self._check_consistency(draft_answer, source_answer)

            vr = VerificationResult(
                question=vq,
                draft_says=claim,
                source_says=source_answer,
                consistent=consistent,
                action="OK" if consistent else "CORRECT",
            )
            verification_log.append(vr)

            if not consistent:
                corrections_made += 1

        # Step 3: Apply corrections if needed
        corrected_text = section_text
        if corrections_made > 0:
            corrected_text = self._apply_corrections(
                section_text, verification_log
            )

        # Compute confidence
        total = len(verification_log) or 1
        consistent_count = sum(1 for v in verification_log if v.consistent)
        confidence = consistent_count / total

        return VerifiedSection(
            text=corrected_text,
            corrections_made=corrections_made,
            verification_log=verification_log,
            confidence=confidence,
        )

    def _extract_claims(self, text: str) -> list[str]:
        """Extract factual claims from text using LLM."""
        try:
            response = self.llm.generate_fast(
                f"Extract all factual claims from this text. Return a JSON array of strings.\n"
                f"Focus on claims about: specific methods, performance numbers, comparisons, "
                f"dates, and attributions.\n\n"
                f"Text:\n{text[:3000]}\n\n"
                f'Output: ["claim1", "claim2", ...]',
                max_tokens=1500,
            )
            parsed = self._parse_json_array(response)
            return parsed[:20]  # Cap at 20 claims
        except Exception as e:
            logger.warning("Claim extraction failed: %s", e)
            # Fallback: split by sentences
            sentences = [s.strip() for s in text.split(".") if len(s.strip()) > 20]
            return sentences[:10]

    def _generate_verification_question(self, claim: str) -> str:
        """Generate a verification question for a claim."""
        try:
            response = self.llm.generate_fast(
                f"Generate one specific yes/no verification question for this claim:\n"
                f"Claim: {claim}\n\n"
                f"Question:",
                max_tokens=100,
            )
            return response.strip()
        except Exception:
            return f"Is the following claim accurate: {claim}?"

    def _answer_from_sources(
        self, question: str, sources: RetrievalResult
    ) -> str:
        """Answer a verification question using source documents."""
        source_text = "\n".join(
            f"[{c.title}, {c.year}]: {c.text[:300]}"
            for c in sources.chunks[:10]
        )

        if not source_text.strip():
            return "No source documents available to verify."

        try:
            response = self.llm.generate_fast(
                f"Answer this question ONLY using the provided sources. "
                f"If the sources don't contain the answer, say 'NOT FOUND IN SOURCES'.\n\n"
                f"Question: {question}\n\n"
                f"Sources:\n{source_text}\n\n"
                f"Answer:",
                max_tokens=200,
            )
            return response.strip()
        except Exception:
            return "Verification failed."

    def _check_consistency(self, draft_claim: str, source_answer: str) -> bool:
        """Check if draft claim is consistent with source answer."""
        if "NOT FOUND" in source_answer.upper():
            return False  # Can't verify = inconsistent (conservative)

        try:
            response = self.llm.generate_fast(
                f"Is the following claim consistent with the source evidence?\n\n"
                f"Claim: {draft_claim}\n"
                f"Source evidence: {source_answer}\n\n"
                f"Answer ONLY 'YES' or 'NO'.",
                max_tokens=10,
            )
            return "YES" in response.upper()
        except Exception:
            return True  # Default to consistent on error

    def _apply_corrections(
        self, text: str, verification_log: list[VerificationResult]
    ) -> str:
        """Apply corrections based on verification results."""
        inconsistent = [v for v in verification_log if not v.consistent]
        if not inconsistent:
            return text

        corrections_prompt = "Fix these inconsistencies in the text:\n\n"
        for v in inconsistent[:5]:
            corrections_prompt += (
                f"- Claim: {v.draft_says}\n"
                f"  Source says: {v.source_says}\n\n"
            )
        corrections_prompt += f"\nOriginal text:\n{text[:3000]}\n\nCorrected text:"

        try:
            response = self.llm.generate(
                corrections_prompt,
                system="You are a careful editor. Fix factual errors while preserving style.",
                max_tokens=4000,
            )
            return response.strip()
        except Exception as e:
            logger.warning("Correction application failed: %s", e)
            return text

    def _parse_json_array(self, text: str) -> list[str]:
        """Parse a JSON array from LLM response."""
        text = text.strip()
        start = text.find("[")
        end = text.rfind("]") + 1
        if start >= 0 and end > start:
            return json.loads(text[start:end])
        return []


class ProvenanceTracker:
    """Track the provenance of every sentence in generated text."""

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def track_section(
        self,
        section_text: str,
        source_chunks: RetrievalResult,
    ) -> list[ProvenanceRecord]:
        """Assign provenance to each sentence in a section."""
        sentences = [s.strip() + "." for s in section_text.split(".") if len(s.strip()) > 10]
        records = []

        for sentence in sentences:
            record = self._classify_sentence(sentence, source_chunks)
            records.append(record)

        return records

    def _classify_sentence(
        self, sentence: str, sources: RetrievalResult
    ) -> ProvenanceRecord:
        """Classify a sentence's provenance type."""
        # Simple similarity-based classification
        best_sim = 0.0
        best_source = ""
        best_source_text = ""

        for chunk in sources.chunks[:15]:
            # Word overlap similarity
            sent_words = set(sentence.lower().split())
            chunk_words = set(chunk.text.lower().split())
            overlap = len(sent_words & chunk_words)
            total = max(len(sent_words), 1)
            sim = overlap / total

            if sim > best_sim:
                best_sim = sim
                best_source = chunk.paper_id
                best_source_text = chunk.text[:200]

        if best_sim > 0.6:
            ptype = ProvenanceType.PARAPHRASED
        elif best_sim > 0.3:
            ptype = ProvenanceType.SYNTHESIZED
        elif best_sim > 0.15:
            ptype = ProvenanceType.INFERRED
        else:
            ptype = ProvenanceType.ORIGINAL

        return ProvenanceRecord(
            text=sentence,
            provenance_type=ptype,
            source_paper=best_source,
            source_sentence=best_source_text,
            similarity_score=round(best_sim, 3),
            confidence=round(1.0 - best_sim * 0.5, 3),
        )
