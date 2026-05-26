"""Execution-based verification of generated temporal samples.

Converts each candidate S-expression to SPARQL via logic_form_util.lisp_to_sparql(),
executes against Virtuoso, and compares predicted vs expected answers with
normalized matching (MID exact, label, date, multi-answer set comparison).
"""

from __future__ import annotations

import re
from typing import List, Optional, Set, Tuple

from .schema import TemporalSample, ValidationStatus


# ── Answer normalization ───────────────────────────────────────────────

def normalize_mid(value: str) -> str:
    """Normalize a Freebase MID to its canonical form."""
    value = value.strip()
    # Strip namespace prefixes
    for prefix in ("http://rdf.freebase.com/ns/", "fb:", "ns:"):
        if value.startswith(prefix):
            value = value[len(prefix):]
    # Remove trailing type annotations
    if "^^" in value:
        value = value.split("^^")[0]
    # Strip quotes
    value = value.strip('"')
    return value


def normalize_date(value: str) -> str:
    """Normalize date values for comparison.

    '2010', '2010-01-01', '2010-1-1' all match each other at year level.
    Returns YYYY-MM-DD or YYYY format.
    """
    value = normalize_mid(value)

    # Extract date parts
    match = re.match(r"(\d{4})(?:-(\d{1,2})(?:-(\d{1,2}))?)?", value)
    if not match:
        return value

    year = match.group(1)
    month = match.group(2)
    day = match.group(3)

    if month and day:
        return f"{year}-{int(month):02d}-{int(day):02d}"
    if month:
        return f"{year}-{int(month):02d}"
    return year


def normalize_label(value: str) -> str:
    """Normalize entity labels for fuzzy comparison."""
    value = normalize_mid(value)
    return value.strip().lower()


def is_mid(value: str) -> bool:
    """Check if a value looks like a Freebase MID (m.XXXXX or g.XXXXX)."""
    cleaned = normalize_mid(value)
    return bool(re.match(r"^[mg]\.[a-zA-Z0-9_]+$", cleaned))


def is_date_like(value: str) -> bool:
    """Check if a value looks like a date."""
    cleaned = normalize_mid(value)
    return bool(re.match(r"^\d{4}(-\d{2}(-\d{2})?)?$", cleaned))


def normalize_answer(value: str) -> str:
    """Normalize a single answer value for comparison."""
    value = normalize_mid(value)
    if is_date_like(value):
        return normalize_date(value)
    if is_mid(value):
        return value
    return normalize_label(value)


def normalize_answer_set(answers: List[str]) -> Set[str]:
    """Normalize a list of answer strings into a set for comparison."""
    return {normalize_answer(a) for a in answers if a and a.strip()}


# ── Answer matching ────────────────────────────────────────────────────

def _mid_match(pred: str, expected: str) -> bool:
    """Exact MID match."""
    return normalize_mid(pred) == normalize_mid(expected)


def _label_match(pred: str, expected: str) -> bool:
    """Case-insensitive label match."""
    return normalize_label(pred) == normalize_label(expected)


def _date_year_match(pred: str, expected: str) -> bool:
    """Match dates at year granularity (2010 == 2010-01-01)."""
    pred_date = normalize_date(pred)
    expected_date = normalize_date(expected)
    if pred_date == expected_date:
        return True
    # Year-only comparison
    pred_year = pred_date[:4] if len(pred_date) >= 4 else ""
    expected_year = expected_date[:4] if len(expected_date) >= 4 else ""
    return pred_year == expected_year and len(pred_year) == 4


def answers_match(
    predicted: List[str],
    expected: List[str],
) -> bool:
    """Check if predicted answers match expected answers.

    Strategies tried in order:
    1. MID exact match (any predicted MID matches any expected MID)
    2. If one side is date-like, date-year match
    3. Label normalized match (case-insensitive, trimmed)
    4. Set intersection non-empty
    """
    if not predicted or not expected:
        return False

    pred_set = normalize_answer_set(predicted)
    exp_set = normalize_answer_set(expected)

    # 1. MID exact intersection
    pred_mids = {a for a in pred_set if is_mid(a)}
    exp_mids = {a for a in exp_set if is_mid(a)}
    if pred_mids and exp_mids and pred_mids & exp_mids:
        return True

    # 2. Date-year match
    pred_dates = {normalize_date(a) for a in predicted if is_date_like(normalize_mid(a))}
    exp_dates = {normalize_date(a) for a in expected if is_date_like(normalize_mid(a))}
    for pd in pred_dates:
        for ed in exp_dates:
            if _date_year_match(pd, ed):
                return True

    # 3. Label intersection
    pred_labels = {normalize_label(a) for a in predicted}
    exp_labels = {normalize_label(a) for a in expected}
    if pred_labels & exp_labels:
        return True

    # 4. Subset or set equality (after normalization)
    if pred_set & exp_set:
        return True

    return False


# ── Sample verification ────────────────────────────────────────────────

def _s_expr_to_sparql(s_expression: str) -> Optional[str]:
    """Convert an S-expression to a SPARQL query string.

    Uses logic_form_util.lisp_to_sparql (lazy import, requires Virtuoso libs).
    """
    if not s_expression or s_expression.lower() == "null":
        return None
    try:
        from executor.logic_form_util import lisp_to_sparql
        sparql = lisp_to_sparql(s_expression)
        if sparql and "SELECT" in sparql.upper():
            return sparql
        return None
    except Exception:
        return None


def verify_sample(
    sample: TemporalSample,
    timeout: int = 30,
) -> Tuple[bool, Optional[List[str]], Optional[str]]:
    """Execute a sample's S-expression and compare results.

    Args:
        sample: TemporalSample to verify.
        timeout: SPARQL query timeout in seconds.

    Returns:
        (passed, predicted_answers, error_message)
        - passed: True if answers match
        - predicted_answers: List of answer strings from execution
        - error_message: None if successful, error description if failed
    """
    # Convert S-expression to SPARQL
    sparql = _s_expr_to_sparql(sample.s_expression)
    if sparql is None:
        return False, None, "S-expression to SPARQL conversion failed"

    # Store SPARQL for provenance
    sample.sparql = sparql

    # Execute
    try:
        from executor.sparql_executor import execute_query
        predicted = execute_query(sparql)
    except Exception as exc:
        return False, None, f"SPARQL execution error: {exc}"

    if not predicted:
        return False, [], "Query returned no results"

    # Compare answers
    expected = list(sample.answers)
    if answers_match(predicted, expected):
        # Update sample with actual results
        sample.answers = predicted
        return True, predicted, None

    return False, predicted, f"Answer mismatch: pred={predicted[:5]}, expected={expected[:5]}"


def verify_samples(
    samples: List[TemporalSample],
    timeout: int = 30,
    verbose: bool = True,
) -> Tuple[List[TemporalSample], List[TemporalSample]]:
    """Verify a list of candidate samples, returning (passed, failed) splits.

    Args:
        samples: Candidate TemporalSamples to verify.
        timeout: SPARQL query timeout per sample.
        verbose: Print progress to stdout.

    Returns:
        (passed_samples, failed_samples) — passed samples have their
        validation_status updated to EXECUTABLE and answers replaced
        with actual KB results.
    """
    passed: List[TemporalSample] = []
    failed: List[TemporalSample] = []

    for idx, sample in enumerate(samples):
        if verbose and (idx + 1) % 50 == 0:
            print(f"  [VERIFY] {idx + 1}/{len(samples)} — {len(passed)} passed, {len(failed)} failed")

        ok, predicted, error = verify_sample(sample, timeout=timeout)

        if ok:
            sample.validation_status = ValidationStatus.EXECUTABLE
            sample.metadata["verified"] = True
            sample.metadata["predicted_answers"] = predicted
            passed.append(sample)
        else:
            sample.validation_status = ValidationStatus.FAILED
            sample.metadata["verified"] = False
            sample.metadata["verification_error"] = error
            if predicted:
                sample.metadata["predicted_answers"] = predicted
            failed.append(sample)

    if verbose:
        total = len(passed) + len(failed)
        pass_rate = len(passed) / total * 100 if total > 0 else 0
        print(f"  [VERIFY DONE] {len(passed)}/{total} passed ({pass_rate:.1f}%)")

    return passed, failed


def summarize_verification(
    passed: List[TemporalSample],
    failed: List[TemporalSample],
) -> dict:
    """Summary statistics for verification results."""
    from collections import Counter

    error_reasons = Counter()
    for sample in failed:
        error = sample.metadata.get("verification_error", "unknown")
        # Classify error type
        if "no results" in error:
            error_reasons["no_results"] += 1
        elif "conversion failed" in error:
            error_reasons["conversion_failed"] += 1
        elif "Answer mismatch" in error:
            error_reasons["answer_mismatch"] += 1
        elif "execution error" in error:
            error_reasons["execution_error"] += 1
        else:
            error_reasons["other"] += 1

    return {
        "total_candidates": len(passed) + len(failed),
        "passed": len(passed),
        "failed": len(failed),
        "pass_rate": round(len(passed) / (len(passed) + len(failed)) * 100, 1)
        if (len(passed) + len(failed)) > 0
        else 0,
        "failure_reasons": dict(sorted(error_reasons.items())),
    }
