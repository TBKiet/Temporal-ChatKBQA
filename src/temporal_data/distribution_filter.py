"""Post-verification distribution filter.

Deduplicates, caps per relation family / reasoning type / date bucket,
and balances the dataset before training export.
"""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from typing import Dict, List, Optional

from .schema import TemporalSample


def _extract_year(date_str: str) -> Optional[int]:
    """Extract year from a date string."""
    if not date_str:
        return None
    try:
        return int(str(date_str)[:4])
    except (ValueError, IndexError):
        return None


def _date_bucket(year: Optional[int], bucket_size: int = 10) -> str:
    """Map a year to a decade bucket string."""
    if year is None:
        return "unknown"
    decade = (year // bucket_size) * bucket_size
    return f"{decade}s"


def deduplicate(samples: List[TemporalSample]) -> List[TemporalSample]:
    """Remove duplicates by (question, s_expression) pair."""
    seen: set = set()
    unique: List[TemporalSample] = []
    for sample in samples:
        key = (sample.question.strip().lower(), sample.s_expression.strip())
        if key in seen:
            continue
        seen.add(key)
        unique.append(sample)
    return unique


def cap_by_family(
    samples: List[TemporalSample],
    max_per_family: int = 500,
    seed: int = 42,
) -> List[TemporalSample]:
    """Cap samples per relation family to avoid over-representation."""
    rng = random.Random(seed)
    by_family: Dict[str, List[TemporalSample]] = defaultdict(list)
    for sample in samples:
        family = sample.metadata.get("fact_relation_family", "unknown")
        by_family[family].append(sample)

    capped: List[TemporalSample] = []
    for family, items in sorted(by_family.items()):
        if len(items) > max_per_family:
            capped.extend(rng.sample(items, max_per_family))
        else:
            capped.extend(items)

    return capped


def cap_by_reasoning_type(
    samples: List[TemporalSample],
    max_per_type: int = 1000,
    seed: int = 42,
) -> List[TemporalSample]:
    """Cap samples per temporal reasoning type."""
    rng = random.Random(seed)
    by_type: Dict[str, List[TemporalSample]] = defaultdict(list)
    for sample in samples:
        ttype = sample.temporal_type.value
        by_type[ttype].append(sample)

    capped: List[TemporalSample] = []
    for ttype, items in sorted(by_type.items()):
        if len(items) > max_per_type:
            capped.extend(rng.sample(items, max_per_type))
        else:
            capped.extend(items)

    return capped


def cap_by_date_bucket(
    samples: List[TemporalSample],
    max_per_bucket: int = 200,
    seed: int = 42,
) -> List[TemporalSample]:
    """Cap samples per decade bucket to avoid temporal bias."""
    rng = random.Random(seed)
    by_bucket: Dict[str, List[TemporalSample]] = defaultdict(list)
    for sample in samples:
        start = sample.metadata.get("temporal_start", "")
        year = _extract_year(start)
        bucket = _date_bucket(year)
        by_bucket[bucket].append(sample)

    capped: List[TemporalSample] = []
    for bucket, items in sorted(by_bucket.items()):
        if len(items) > max_per_bucket:
            capped.extend(rng.sample(items, max_per_bucket))
        else:
            capped.extend(items)

    return capped


def apply_distribution_filter(
    samples: List[TemporalSample],
    max_per_family: int = 500,
    max_per_type: int = 1000,
    max_per_date_bucket: int = 200,
    min_sample_length: int = 10,
    seed: int = 42,
) -> List[TemporalSample]:
    """Apply full distribution filter pipeline.

    Order: deduplicate → cap by family → cap by reasoning type → cap by date bucket.

    Args:
        samples: Verified TemporalSamples.
        max_per_family: Max samples per relation family.
        max_per_type: Max samples per temporal reasoning type.
        max_per_date_bucket: Max samples per decade.
        min_sample_length: Minimum question length (filter short/empty questions).
        seed: Random seed for reproducibility.

    Returns:
        Filtered, balanced list of TemporalSamples.
    """
    # Pre-filter: minimum question length
    filtered = [s for s in samples if len(s.question.strip()) >= min_sample_length]

    # Step 1: Deduplicate
    unique = deduplicate(filtered)
    print(f"  [FILTER] Dedup: {len(filtered)} → {len(unique)}")

    # Step 2: Cap by relation family
    by_family = cap_by_family(unique, max_per_family=max_per_family, seed=seed)
    print(f"  [FILTER] Cap by family (max {max_per_family}): {len(unique)} → {len(by_family)}")

    # Step 3: Cap by reasoning type
    by_type = cap_by_reasoning_type(by_family, max_per_type=max_per_type, seed=seed)
    print(f"  [FILTER] Cap by reasoning type (max {max_per_type}): {len(by_family)} → {len(by_type)}")

    # Step 4: Cap by date bucket
    result = cap_by_date_bucket(by_type, max_per_bucket=max_per_date_bucket, seed=seed)
    print(f"  [FILTER] Cap by date bucket (max {max_per_date_bucket}): {len(by_type)} → {len(result)}")

    return result


def summarize_distribution(samples: List[TemporalSample]) -> dict:
    """Summarize final distribution of samples after filtering."""
    from collections import Counter

    by_type = Counter(s.temporal_type.value for s in samples)
    by_family = Counter(s.metadata.get("fact_relation_family", "unknown") for s in samples)
    by_template = Counter(s.metadata.get("template_family", "unknown") for s in samples)
    by_bucket = Counter(
        _date_bucket(_extract_year(s.metadata.get("temporal_start", "")))
        for s in samples
    )

    return {
        "total_samples": len(samples),
        "by_temporal_type": dict(sorted(by_type.items())),
        "by_relation_family": dict(sorted(by_family.items())),
        "by_template_family": dict(sorted(by_template.items())),
        "by_date_bucket": dict(sorted(by_bucket.items())),
        "unique_questions": len({s.question for s in samples}),
    }
