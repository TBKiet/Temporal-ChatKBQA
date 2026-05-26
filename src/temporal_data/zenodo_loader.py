"""Load temporal facts from Zenodo idirlab/freebases dataset.

Replaces SPARQL mining when Freebase Virtuoso lacks label resolution.
Uses the FB+CVT+REV variant (4,425 relations, 122M entities, 244M triples).

Pipeline:
  1. Load relation2id.txt → {relation_path: numeric_id}
  2. Filter for temporal-relevant relations (from whitelist)
  3. Stream train.txt → extract triples with temporal relation IDs
  4. Resolve entity numeric IDs → MIDs via entity2id.txt (chunked, not full load)
  5. Look up MID → label via entities_id_label.csv
  6. Build MinedTemporalFact records
"""

from __future__ import annotations

import csv
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

from .sparql_miner import MinedTemporalFact

# Handle very long entity labels in CSV
csv.field_size_limit(sys.maxsize)


# ── Temporal relation patterns ────────────────────────────────────────

TEMPORAL_RELATION_PATTERNS: List[str] = [
    # Sports
    "/sports/pro_athlete/teams",
    "/sports/sports_team_roster/position",
    "/sports/sports_team_roster/team",
    "/sports/sports_team/roster",
    # Film
    "/film/film/directed_by",
    "/film/film/starring",
    "/film/performance/actor",
    "/film/director/film",
    # Music
    "/music/artist/album",
    "/music/artist/track",
    "/music/group_member/membership",
    # People
    "/people/marriage/spouse",
    "/people/person/date_of_birth",
    "/people/person/date_of_death",
    "/people/person/gender",
    "/people/deceased_person/date_of_death",
    # Books
    "/book/author/works_written",
    "/book/written_work/author",
    # Awards
    "/award/award_nominee/award_nominations",
    "/award/award_nomination/award_nominee",
    # Government
    "/government/government_position_held/office_holder",
    "/government/politician/government_positions_held",
    # Organization
    "/organization/leadership/person",
    "/organization/organization/leadership",
    # Additional CVT-related
    "/sports/sports_team_roster/position",
    "/film/film/release_date_s",
    "/music/album/release_date",
]


# ── Relation families mapping (Zenodo path → whitelist family) ────────

ZENODO_TO_FAMILY: Dict[str, str] = {
    "/sports/pro_athlete/teams": "sports_team",
    "/sports/sports_team_roster/position": "sports_team",
    "/sports/sports_team_roster/team": "sports_team",
    "/sports/sports_team/roster": "sports_team",
    "/film/film/directed_by": "film_directed_by",
    "/film/film/starring": "film_starring",
    "/film/performance/actor": "film_starring",
    "/film/director/film": "film_directed_by",
    "/music/artist/album": "music_album",
    "/music/artist/track": "music_album",
    "/people/marriage/spouse": "marriage",
    "/people/person/date_of_birth": "date_of_birth",
    "/people/person/date_of_death": "date_of_death",
    "/people/deceased_person/date_of_death": "date_of_death",
    "/book/author/works_written": "book_author",
    "/book/written_work/author": "book_author",
    "/award/award_nominee/award_nominations": "award_nomination",
    "/award/award_nomination/award_nominee": "award_nomination",
    "/government/government_position_held/office_holder": "government_position",
    "/government/politician/government_positions_held": "government_position",
    "/organization/leadership/person": "organization_leadership",
    "/organization/organization/leadership": "organization_leadership",
}


def _zenodo_path_to_dotted(zenodo_path: str) -> str:
    """Convert /sports/pro_athlete/teams → sports.pro_athlete.teams"""
    return zenodo_path.strip("/").replace("/", ".")


def _clean_mid(mid: str) -> str:
    """Convert /m/05yk8w2 → m.05yk8w2, /g/112yfxf28 → g.112yfxf28"""
    mid = mid.strip()
    if mid.startswith("/") and mid.count("/") >= 2:
        parts = mid[1:].split("/")
        mid = parts[0] + "." + parts[1]
    return mid


def load_relation_map(relation2id_path: str) -> Dict[str, int]:
    """Load relation path → numeric ID mapping. Only 4,425 entries."""
    rel_map: Dict[str, int] = {}
    with open(relation2id_path, encoding="utf8") as fh:
        for line in fh:
            parts = line.strip().split(",")
            if len(parts) == 2:
                rel_path = parts[0].strip()
                rel_id = int(parts[1].strip())
                rel_map[rel_path] = rel_id
    return rel_map


def select_temporal_relation_ids(rel_map: Dict[str, int]) -> Dict[str, Tuple[int, str]]:
    """From the full relation map, select temporal-relevant ones.

    Returns: {zenodo_path: (numeric_id, family_name)}
    """
    selected: Dict[str, Tuple[int, str]] = {}
    for pattern in TEMPORAL_RELATION_PATTERNS:
        if pattern in rel_map:
            family = ZENODO_TO_FAMILY.get(pattern, "unknown")
            selected[pattern] = (rel_map[pattern], family)
    return selected


def load_entity_labels(
    entities_id_label_path: str,
    target_entity_ids: Set[int],
    max_entities: int = 500000,
) -> Dict[int, str]:
    """Load labels for specific entity numeric IDs.

    entities_id_label.csv format: /g/112yfxf28, Ootaki\\, Yuuko, 2149
    Note: the numeric ID in this file is the author-assigned ID,
    NOT the sequential ID from entity2id.txt. We need both mappings.
    """
    labels: Dict[int, str] = {}
    with open(entities_id_label_path, encoding="utf8") as fh:
        reader = csv.reader(fh)
        for row in reader:
            if len(row) >= 3:
                try:
                    eid = int(row[2].strip())
                except ValueError:
                    continue
                if eid in target_entity_ids or len(labels) < max_entities:
                    labels[eid] = row[1].strip()
                if len(labels) >= max_entities and not target_entity_ids:
                    break
    return labels


def load_entity_id_to_numeric(
    entity2id_path: str,
    entity_ids_of_interest: Set[int],
) -> Dict[int, int]:
    """Build mapping from ID in triple files to entity numeric ID for labels.

    entity2id.txt: /g/112yf8_qt, 0 → MID, sequential_id
    We need: sequential_id → entity_label_numeric_id

    This requires cross-referencing MIDs with entities_id_label.csv.
    """
    # entity2id maps: sequential_id → MID
    seq_to_mid: Dict[int, str] = {}
    with open(entity2id_path, encoding="utf8") as fh:
        for line in fh:
            parts = line.strip().split(",")
            if len(parts) == 2:
                try:
                    seq_id = int(parts[1].strip())
                except ValueError:
                    continue
                if seq_id in entity_ids_of_interest:
                    seq_to_mid[seq_id] = parts[0].strip()
                if len(seq_to_mid) >= len(entity_ids_of_interest):
                    break
    return seq_to_mid


def extract_temporal_triples(
    train_path: str,
    temporal_rel_ids: Set[int],
    max_triples_per_rel: int = 5000,
) -> Dict[int, List[Tuple[int, int]]]:
    """Stream train.txt and extract triples with temporal relations.

    train.txt format: head_id, relation_id, tail_id

    Returns: {relation_id: [(head_id, tail_id), ...]}
    """
    triples: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
    counts: Dict[int, int] = defaultdict(int)

    with open(train_path, encoding="utf8") as fh:
        for line in fh:
            parts = line.strip().split(",")
            if len(parts) != 3:
                continue
            try:
                head_id = int(parts[0].strip())
                rel_id = int(parts[1].strip())
                tail_id = int(parts[2].strip())
            except ValueError:
                continue

            if rel_id in temporal_rel_ids and counts[rel_id] < max_triples_per_rel:
                triples[rel_id].append((head_id, tail_id))
                counts[rel_id] += 1

            # Early exit if all relations reached their cap
            if all(c >= max_triples_per_rel for c in counts.values()) and len(counts) == len(temporal_rel_ids):
                break

    return dict(triples)


def mine_facts_from_zenodo(
    data_dir: str,
    variant: str = "FB+CVT+REV",
    max_triples_per_rel: int = 200,
    max_facts_total: int = 10000,
) -> List["MinedTemporalFact"]:
    """Main entry point: mine temporal facts from Zenodo dataset.

    Args:
        data_dir: Path to extracted Zenodo dataset root.
        variant: Dataset variant (FB+CVT+REV is recommended).
        max_triples_per_rel: Max triples to extract per relation.
        max_facts_total: Absolute cap on total facts returned.

    Returns:
        List of MinedTemporalFact records with labels resolved.
    """
    metadata_dir = os.path.join(data_dir, "idirlab-freebases", "Metadata")
    variant_dir = os.path.join(data_dir, "idirlab-freebases", variant)

    print(f"Loading relations from {variant} variant...")
    rel_map = load_relation_map(os.path.join(variant_dir, "relation2id.txt"))
    print(f"  {len(rel_map)} total relations")

    temporal_rels = select_temporal_relation_ids(rel_map)
    temporal_rel_ids = {rid for rid, _ in temporal_rels.values()}
    print(f"  {len(temporal_rels)} temporal relations selected")

    # Build relation_id → (path, family) lookup
    rel_id_to_info: Dict[int, Tuple[str, str]] = {}
    for path, (rid, family) in temporal_rels.items():
        rel_id_to_info[rid] = (path, family)

    print(f"Scanning train.txt for temporal triples (max {max_triples_per_rel}/rel)...")
    triple_data = extract_temporal_triples(
        os.path.join(variant_dir, "train.txt"),
        temporal_rel_ids,
        max_triples_per_rel=max_triples_per_rel,
    )
    total_triples = sum(len(v) for v in triple_data.values())
    print(f"  {total_triples} triples found across {len(triple_data)} relations")

    # Collect entity IDs that need label resolution
    needed_entity_ids: Set[int] = set()
    for rel_id, tuples in triple_data.items():
        for head_id, tail_id in tuples:
            needed_entity_ids.add(head_id)
            needed_entity_ids.add(tail_id)
    print(f"  {len(needed_entity_ids)} unique entity IDs to resolve")

    # Load entity MID mappings for needed IDs
    print("Loading entity ID → MID mappings...")
    seq_to_mid = load_entity_id_to_numeric(
        os.path.join(variant_dir, "entity2id.txt"),
        needed_entity_ids,
    )
    print(f"  {len(seq_to_mid)} MIDs resolved")

    # Build MID → label from entities_id_label.csv
    print("Loading entity labels...")
    mid_to_label: Dict[str, str] = {}
    label_file = os.path.join(metadata_dir, "entities_id_label.csv")
    with open(label_file, encoding="utf8") as fh:
        reader = csv.reader(fh)
        for row in reader:
            if len(row) >= 2:
                mid = row[0].strip()
                label = row[1].strip()
                mid_to_label[mid] = label
    print(f"  {len(mid_to_label)} entity labels loaded")

    # Build facts
    print("Building MinedTemporalFact records...")
    facts: List[MinedTemporalFact] = []
    import hashlib

    for rel_id, tuples in triple_data.items():
        if rel_id not in rel_id_to_info:
            continue
        rel_path, family = rel_id_to_info[rel_id]
        dotted_rel = _zenodo_path_to_dotted(rel_path)

        for head_id, tail_id in tuples:
            if len(facts) >= max_facts_total:
                break

            head_mid_raw = seq_to_mid.get(head_id, "")
            tail_mid_raw = seq_to_mid.get(tail_id, "")
            if not head_mid_raw or not tail_mid_raw:
                continue

            head_label = mid_to_label.get(head_mid_raw, "")
            tail_label = mid_to_label.get(tail_mid_raw, "")
            if not head_label or not tail_label:
                continue

            # Clean MID only for S-expression output
            head_mid = _clean_mid(head_mid_raw)
            tail_mid = _clean_mid(tail_mid_raw)

            row_hash = hashlib.sha256(
                f"{head_mid_raw}|{dotted_rel}|{tail_mid_raw}".encode()
            ).hexdigest()[:12]

            # Derive role hint from LAST part of relation path
            # e.g., /book/author/works_written → "written work"
            #       /film/film/directed_by → "director"
            #       /music/artist/album → "album"
            rel_parts = rel_path.strip("/").split("/")
            raw_role = rel_parts[-1] if rel_parts else "entity"
            role_hint = raw_role.replace("_", " ")

            fact = MinedTemporalFact(
                fact_id=f"zenodo-{family}-{row_hash}",
                fact_relation_family=family,
                topic_mid=head_mid,
                topic_label=head_label,
                anchor_relation=dotted_rel,
                answer_relation=dotted_rel,
                answer_mid=tail_mid,
                answer_label=tail_label,
                temporal_start="",
                temporal_end="",
                ordering_value="",
                retrieved_from_relation=dotted_rel,
                source_query_name=f"zenodo_{variant}",
                supporting_row_hash=row_hash,
                metadata={
                    "answer_type": "entity",
                    "zenodo_rel_id": rel_id,
                    "zenodo_rel_path": rel_path,
                    "role_hint": role_hint,
                },
            )
            facts.append(fact)

        if len(facts) >= max_facts_total:
            break

    print(f"  {len(facts)} facts built with labels")
    return facts
