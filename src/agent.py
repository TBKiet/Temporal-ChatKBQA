"""Agentic routing component for Temporal KBQA.

TemporalQuestionAgent implements multi-step reasoning:
  Step 1 — Detect temporal signals in question
  Step 2 — Route to temporal or standard KBQA pipeline
  Step 3 — Iterative refinement on empty results (relax temporal constraints)
  Step 4 — Return answer with provenance (SPARQL + constraint used)
"""

import re
import sys
import os
from typing import Optional

# Add project root to path so imports work when called from src/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


TEMPORAL_SIGNALS = [
    r'\bbefore\b', r'\bafter\b', r'\bduring\b', r'\bwhen\b',
    r'\bfirst\b', r'\blast\b', r'\blatest\b', r'\bearli(?:est|er)\b',
    r'\bmost recent\b', r'\bat the time\b', r'\bby \d{4}\b',
    r'\bin \d{4}\b', r'\bsince \d{4}\b', r'\buntil \d{4}\b',
    r'\d{4}',
]

_TEMPORAL_PATTERN = re.compile(
    '|'.join(TEMPORAL_SIGNALS), re.IGNORECASE
)


def detect_temporal_signals(question: str) -> list[str]:
    """Return list of matched temporal signal strings in question."""
    return _TEMPORAL_PATTERN.findall(question)


class TemporalQuestionAgent:
    """Agent that routes questions to temporal or standard KBQA pipeline.

    Supports agentic behaviours required by the assignment:
    - Multi-step reasoning (detect → route → execute → refine)
    - Tool usage (Freebase SPARQL via ODBC)
    - Decision-making based on intermediate outputs (retry on empty answer)
    """

    def __init__(self, pipeline, max_retries: int = 2):
        """
        Args:
            pipeline: An instance of TemporalKBQAPipeline (src/pipeline.py).
            max_retries: Number of refinement attempts when answer is empty.
        """
        self.pipeline = pipeline
        self.max_retries = max_retries

    def run(self, question: str, mode: Optional[str] = None) -> dict:
        """Main agent entrypoint. Returns answer dict with provenance.

        Args:
            question: Natural language question.
            mode: 'temporal', 'standard', or None (auto-detect from signals).

        Returns:
            {
                "question": str,
                "answer": list[str],
                "is_temporal": bool,
                "temporal_signals": list[str],
                "sparql_used": str | None,
                "temporal_constraint": str | None,
                "reasoning_steps": list[str],
                "retries": int,
            }
        """
        steps = []

        # Step 1: Temporal signal detection
        signals = detect_temporal_signals(question)
        auto_is_temporal = len(signals) > 0

        # Respect explicit mode override, otherwise auto-detect
        if mode == 'temporal':
            is_temporal = True
            steps.append(
                f"Step 1 — Temporal mode forced by user. "
                f"Auto-detected signals: {signals}"
            )
        elif mode == 'standard':
            is_temporal = False
            steps.append(
                f"Step 1 — Standard mode forced by user. "
                f"Ignored signals: {signals}"
            )
        else:
            is_temporal = auto_is_temporal
            steps.append(
                f"Step 1 — Detected {'temporal' if is_temporal else 'non-temporal'} question. "
                f"Signals: {signals}"
            )

        # Step 2: Route
        pipeline_mode = 'temporal' if is_temporal else 'standard'
        steps.append(f"Step 2 — Routing to {pipeline_mode} pipeline.")

        # Step 3: Execute with optional retry
        answer = []
        sparql_used = None
        temporal_constraint = None
        retries = 0

        result = self.pipeline.answer(question, mode=pipeline_mode)
        answer = result.get('answer', [])
        sparql_used = result.get('sparql')
        temporal_constraint = result.get('temporal_constraint')
        steps.append(f"Step 3 — First execution: {len(answer)} answer(s) found.")

        # Iterative refinement: if no answer found, relax temporal constraint
        while not answer and retries < self.max_retries:
            retries += 1
            steps.append(
                f"Step 3.{retries} — No answer found. "
                f"Retrying with relaxed temporal constraints (attempt {retries})."
            )
            result = self.pipeline.answer(
                question, mode='standard', relaxed=True
            )
            answer = result.get('answer', [])
            sparql_used = result.get('sparql')
            temporal_constraint = result.get('temporal_constraint')

        # Step 4: Return answer with provenance
        steps.append(
            f"Step 4 — Final answer: {answer}. "
            f"Constraint: {temporal_constraint}."
        )

        return {
            'question': question,
            'answer': answer,
            'is_temporal': is_temporal,
            'temporal_signals': signals,
            'sparql_used': sparql_used,
            'temporal_constraint': temporal_constraint,
            'reasoning_steps': steps,
            'retries': retries,
        }


def demo_agent_routing(questions: Optional[list] = None):
    """Demonstrate agent routing decisions without requiring a live pipeline."""
    if questions is None:
        questions = [
            "Who was the US president before Obama?",
            "What is the capital of France?",
            "Who held the position of US president during World War II?",
            "What movies did Hitchcock direct in 1960?",
            "Name the first wife of Henry VIII.",
            "Which country won the most gold medals at the 2008 Olympics?",
        ]

    print("=" * 60)
    print("Agent Routing Demo (signal detection only)")
    print("=" * 60)
    for q in questions:
        signals = detect_temporal_signals(q)
        is_temporal = len(signals) > 0
        route = "TEMPORAL pipeline" if is_temporal else "STANDARD pipeline"
        print(f"\nQ: {q}")
        print(f"   Signals found: {signals}")
        print(f"   -> Route: {route}")
    print("=" * 60)


if __name__ == '__main__':
    demo_agent_routing()
