"""Direct SPARQL fact mining against Freebase Virtuoso.

Queries CVT nodes per relation family declared in the whitelist config.
Produces enriched MinedTemporalFact records ready for template generation.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

import yaml

# Lazy imports: execute_query, get_label, SPARQLWrapper depend on a running
# Virtuoso instance. Imported inside functions that actually use them.


@dataclass
class MinedTemporalFact:
    """Normalized temporal fact from Freebase SPARQL mining."""

    fact_id: str
    fact_relation_family: str
    topic_mid: str
    topic_label: str
    anchor_relation: str
    answer_relation: str
    answer_mid: str
    answer_label: str
    temporal_start: str
    temporal_end: str
    ordering_value: str
    retrieved_from_relation: str
    source_query_name: str
    supporting_row_hash: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ── SPARQL query builders per template family ──────────────────────────

def _build_cvt_date_range_query(family_cfg: dict, limit: int) -> str:
    """SPARQL for CVT relations with start/end date range (e.g. sports teams, marriage)."""
    anchor = family_cfg["anchor_relation"]
    answer = family_cfg["answer_relation"]
    temporal = family_cfg["temporal_field"]
    second = family_cfg.get("second_temporal", temporal.replace(".from", ".to") if ".from" in temporal else temporal)

    return f"""
PREFIX ns: <http://rdf.freebase.com/ns/>
SELECT DISTINCT ?topic ?topicLabel ?answer ?answerLabel ?startDate ?endDate
WHERE {{
  ?topic ns:{anchor} ?cvt .
  ?cvt ns:{answer} ?answer .
  ?cvt ns:{temporal} ?startDate .
  OPTIONAL {{ ?cvt ns:{second} ?endDate . }}
  ?topic rdfs:label ?topicLabel .
  ?answer rdfs:label ?answerLabel .
  FILTER (lang(?topicLabel) = 'en')
  FILTER (lang(?answerLabel) = 'en')
  FILTER (?topic != ?answer)
}}
LIMIT {limit}
"""


def _build_cvt_date_point_query(family_cfg: dict, limit: int) -> str:
    """SPARQL for CVT relations with a single date/ordering property (e.g. film starring, albums)."""
    anchor = family_cfg["anchor_relation"]
    answer = family_cfg["answer_relation"]
    temporal = family_cfg["temporal_field"]

    return f"""
PREFIX ns: <http://rdf.freebase.com/ns/>
SELECT DISTINCT ?topic ?topicLabel ?answer ?answerLabel ?temporalValue
WHERE {{
  ?topic ns:{anchor} ?cvt .
  ?cvt ns:{answer} ?answer .
  ?cvt ns:{temporal} ?temporalValue .
  ?topic rdfs:label ?topicLabel .
  ?answer rdfs:label ?answerLabel .
  FILTER (lang(?topicLabel) = 'en')
  FILTER (lang(?answerLabel) = 'en')
  FILTER (?topic != ?answer)
}}
LIMIT {limit}
"""


QUERY_BUILDERS = {
    "date_range": _build_cvt_date_range_query,
    "date_point": _build_cvt_date_point_query,
    "ordering": _build_cvt_date_point_query,  # same pattern, different interpretation
}


def _normalize_mid(raw: str) -> str:
    """Strip namespace prefixes from a Freebase MID."""
    for prefix in ("http://rdf.freebase.com/ns/", "fb:", "ns:"):
        if raw.startswith(prefix):
            return raw[len(prefix):]
    return raw


def _normalize_date(raw: str) -> str:
    """Extract year or date from various Freebase date literal formats."""
    if raw.startswith('"'):
        # "2005-12-31"^^xsd:dateTime or "2005"
        raw = raw.split('"')[1] if '"' in raw else raw
    # Return just the date part (year or year-month-day)
    return raw[:10] if len(raw) >= 10 else raw


def _hash_row(*fields: str) -> str:
    """Deterministic hash for dedup and provenance."""
    return hashlib.sha256("|".join(fields).encode()).hexdigest()[:12]


def _extract_year(date_str: str) -> Optional[int]:
    """Extract year from a date string, returning None if unparseable."""
    if not date_str:
        return None
    try:
        year_str = date_str[:4]
        year = int(year_str)
        if 1000 <= year <= 2100:
            return year
    except (ValueError, IndexError):
        pass
    return None


def _parse_sparql_row(row: dict, family_cfg: dict, query_name: str) -> Optional[MinedTemporalFact]:
    """Parse a single SPARQL result row into a MinedTemporalFact."""
    topic_raw = row.get("topic", {}).get("value", "")
    answer_raw = row.get("answer", {}).get("value", "")
    if not topic_raw or not answer_raw:
        return None

    topic_mid = _normalize_mid(topic_raw)
    answer_mid = _normalize_mid(answer_raw)
    topic_label = row.get("topicLabel", {}).get("value", "")
    answer_label = row.get("answerLabel", {}).get("value", "")

    start_raw = row.get("startDate", row.get("temporalValue", {})).get("value", "")
    end_raw = row.get("endDate", {}).get("value", "")

    temporal_start = _normalize_date(start_raw) if start_raw else ""
    temporal_end = _normalize_date(end_raw) if end_raw else ""

    row_hash = _hash_row(topic_mid, answer_mid, temporal_start, temporal_end, family_cfg["anchor_relation"])

    fact_id = f"mined-{family_cfg['family']}-{row_hash}"

    return MinedTemporalFact(
        fact_id=fact_id,
        fact_relation_family=family_cfg["family"],
        topic_mid=topic_mid,
        topic_label=topic_label,
        anchor_relation=family_cfg["anchor_relation"],
        answer_relation=family_cfg["answer_relation"],
        answer_mid=answer_mid,
        answer_label=answer_label,
        temporal_start=temporal_start,
        temporal_end=temporal_end,
        ordering_value="",
        retrieved_from_relation=family_cfg["temporal_field"],
        source_query_name=query_name,
        supporting_row_hash=row_hash,
        metadata={
            "answer_type": family_cfg.get("answer_type", "entity"),
            "temporal_field_type": family_cfg.get("temporal_field_type"),
        },
    )


def load_whitelist(path: str) -> dict:
    """Load relation whitelist from YAML config file."""
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def mine_temporal_facts(
    whitelist_path: str,
    limit_per_family: Optional[int] = None,
    family_filter: Optional[List[str]] = None,
) -> List[MinedTemporalFact]:
    """Mine temporal facts from Freebase Virtuoso for all whitelisted relation families.

    Args:
        whitelist_path: Path to relation_whitelist.yaml.
        limit_per_family: Override max facts per family (default: from config or 200).
        family_filter: Optional list of family names to restrict mining to.

    Returns:
        List of enriched MinedTemporalFact records.
    """
    config = load_whitelist(whitelist_path)
    families = config["families"]
    global_limits = config.get("limits", {})

    if family_filter:
        families = [f for f in families if f["family"] in family_filter]

    default_limit = limit_per_family or global_limits.get("max_facts_per_family", 200)
    all_facts: List[MinedTemporalFact] = []
    seen_hashes: set = set()

    for family_cfg in families:
        family_name = family_cfg["family"]
        temporal_type = family_cfg.get("temporal_field_type", "date_point")
        query_builder = QUERY_BUILDERS.get(temporal_type)
        if query_builder is None:
            print(f"  [SKIP] {family_name}: unknown temporal_field_type={temporal_type}")
            continue

        query = query_builder(family_cfg, default_limit)
        query_name = f"mine_{family_name}"
        print(f"  [MINING] {family_name} (anchor={family_cfg['anchor_relation']}, limit={default_limit})")

        # execute_query returns single-variable results — use mine_facts_raw for multi-var
        print(f"  [WARN] {family_name}: execute_query is single-variable. Use mine_facts_raw().")
        print(f"  [DONE] {family_name}: 0 facts (need mine_facts_raw for multi-var queries)")

    return all_facts


def mine_facts_raw(
    whitelist_path: str,
    limit_per_family: Optional[int] = None,
    family_filter: Optional[List[str]] = None,
) -> List[MinedTemporalFact]:
    """Mine temporal facts using raw SPARQLWrapper for multi-variable results.

    This is the primary mining entry point. Use on Vast.ai where Virtuoso is running.
    """
    from SPARQLWrapper import SPARQLWrapper, JSON
    from config import FREEBASE_SPARQL_WRAPPER_URL

    config = load_whitelist(whitelist_path)
    families = config["families"]
    global_limits = config.get("limits", {})
    date_min = global_limits.get("sample_date_min_year", 1800)
    date_max = global_limits.get("sample_date_max_year", 2025)

    if family_filter:
        families = [f for f in families if f["family"] in family_filter]

    default_limit = limit_per_family or global_limits.get("max_facts_per_family", 200)

    sparql = SPARQLWrapper(FREEBASE_SPARQL_WRAPPER_URL)
    sparql.setReturnFormat(JSON)
    sparql.setTimeout(120)

    all_facts: List[MinedTemporalFact] = []
    seen_hashes: set = set()

    for family_cfg in families:
        family_name = family_cfg["family"]
        temporal_type = family_cfg.get("temporal_field_type", "date_point")
        query_builder = QUERY_BUILDERS.get(temporal_type)
        if query_builder is None:
            print(f"  [SKIP] {family_name}: unknown temporal_field_type={temporal_type}")
            continue

        query = query_builder(family_cfg, default_limit)
        query_name = f"mine_{family_name}"
        print(f"  [MINING] {family_name} (anchor={family_cfg['anchor_relation']}, limit={default_limit})")

        try:
            sparql.setQuery(query)
            raw_results = sparql.query().convert()
        except Exception as exc:
            print(f"  [FAIL] {family_name}: SPARQL error — {exc}")
            continue

        family_facts = 0
        bindings = raw_results.get("results", {}).get("bindings", [])
        for binding in bindings:
            fact = _parse_sparql_row(binding, family_cfg, query_name)
            if fact is None:
                continue
            if fact.supporting_row_hash in seen_hashes:
                continue

            # Year bounds filter
            start_year = _extract_year(fact.temporal_start)
            end_year = _extract_year(fact.temporal_end)
            if start_year and (start_year < date_min or start_year > date_max):
                continue
            if end_year and (end_year < date_min or end_year > date_max):
                continue

            # Skip facts with empty labels
            if not fact.topic_label.strip() or not fact.answer_label.strip():
                continue

            seen_hashes.add(fact.supporting_row_hash)
            all_facts.append(fact)
            family_facts += 1

        print(f"  [DONE] {family_name}: {family_facts} facts mined, {len(all_facts)} total")

    return all_facts


def summarize_mined_facts(facts: List[MinedTemporalFact]) -> dict:
    """Generate summary statistics for mined facts."""
    from collections import Counter

    by_family = Counter(f.fact_relation_family for f in facts)
    by_answer_type = Counter(f.metadata.get("answer_type", "unknown") for f in facts)
    with_labels = sum(1 for f in facts if f.topic_label and f.answer_label)
    with_dates = sum(1 for f in facts if f.temporal_start)

    return {
        "total_facts": len(facts),
        "by_family": dict(sorted(by_family.items())),
        "by_answer_type": dict(sorted(by_answer_type.items())),
        "with_labels": with_labels,
        "with_temporal_start": with_dates,
        "with_temporal_end": sum(1 for f in facts if f.temporal_end),
        "unique_topics": len({f.topic_mid for f in facts}),
        "unique_answers": len({f.answer_mid for f in facts}),
        "sample_facts": [
            {
                "fact_id": f.fact_id,
                "family": f.fact_relation_family,
                "topic": f.topic_label,
                "answer": f.answer_label,
                "start": f.temporal_start,
                "end": f.temporal_end,
            }
            for f in facts[:10]
        ],
    }
