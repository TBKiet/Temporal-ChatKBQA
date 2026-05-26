"""Temporal KBQA — shared constants and instructions.

These are referenced by both the data preparation (process_NQ.py)
and inference pipeline (src/pipeline.py) to ensure consistent prompts.
"""

# Instruction used during LLM fine-tuning for temporal questions
TEMPORAL_INSTRUCTION = (
    "Generate a Logical Form query for this temporal question, "
    "using TC operators for date constraints and ARGMAX/ARGMIN for ordinal constraints.\n"
)

# Instruction used for standard (non-temporal) questions
STANDARD_INSTRUCTION = (
    "Generate a Logical Form query that retrieves the information "
    "corresponding to the given question.\n"
)
