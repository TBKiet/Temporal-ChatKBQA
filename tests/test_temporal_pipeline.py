"""Tests for the new T-ChatKBQA temporal data pipeline.

Covers: sparql_miner, template_bank, verifier, distribution_filter.
All tests run offline (no Virtuoso required) using fixture data.
"""

import json
import os
import tempfile
import unittest
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.temporal_data.schema import (
    TemporalDataSource,
    TemporalQuestionType,
    TemporalSample,
    ValidationStatus,
)
from src.temporal_data.sparql_miner import (
    MinedTemporalFact,
    _normalize_mid,
    _normalize_date,
    _hash_row,
    summarize_mined_facts,
)
from src.temporal_data.template_bank import (
    generate_candidates,
    summarize_candidates,
    TEMPLATES,
)
from src.temporal_data.verifier import (
    normalize_mid,
    normalize_date,
    normalize_label,
    normalize_answer,
    normalize_answer_set,
    is_mid,
    is_date_like,
    answers_match,
)
from src.temporal_data.distribution_filter import (
    deduplicate,
    cap_by_family,
    cap_by_reasoning_type,
    apply_distribution_filter,
    summarize_distribution,
)
from src.temporal_data.training import (
    sample_to_training_example,
    build_temporal_training_examples,
)


# ── Fixtures ───────────────────────────────────────────────────────────

def _make_fact(**overrides) -> MinedTemporalFact:
    """Build a minimal MinedTemporalFact for testing."""
    defaults = {
        "fact_id": "test-fact-1",
        "fact_relation_family": "sports_team",
        "topic_mid": "m.02gjb",
        "topic_label": "LeBron James",
        "anchor_relation": "sports.pro_athlete.teams",
        "answer_relation": "sports.pro_athlete.teams.office_holder",
        "answer_mid": "m.0jmc7",
        "answer_label": "Los Angeles Lakers",
        "temporal_start": "2018",
        "temporal_end": "2024",
        "ordering_value": "",
        "retrieved_from_relation": "sports.pro_athlete.teams.from",
        "source_query_name": "mine_sports_team",
        "supporting_row_hash": "abc123def456",
        "metadata": {"answer_type": "entity", "temporal_field_type": "date_range"},
    }
    defaults.update(overrides)
    return MinedTemporalFact(**defaults)


def _make_sample(**overrides) -> TemporalSample:
    """Build a minimal TemporalSample for testing."""
    defaults = {
        "sample_id": "test-sample-1",
        "question": "Who is the team of LeBron James?",
        "split": "train",
        "source": TemporalDataSource.SYNTHETIC,
        "temporal_type": TemporalQuestionType.DURING,
        "temporal_signal": "during",
        "topic_entity_mid": "m.02gjb",
        "s_expression": "(JOIN (R sports.pro_athlete.teams.office_holder) m.02gjb)",
        "sparql": "SELECT ?x WHERE { ?x ?p ?o }",
        "answers": ["m.0jmc7"],
        "validation_status": ValidationStatus.RAW,
        "source_dataset": "SyntheticTemporal",
        "metadata": {
            "template_family": "during",
            "fact_relation_family": "sports_team",
            "temporal_start": "2018",
        },
    }
    defaults.update(overrides)
    return TemporalSample(**defaults)


SAMPLE_WHITELIST = {
    "families": [
        {
            "family": "sports_team",
            "anchor_relation": "sports.pro_athlete.teams",
            "answer_relation": "sports.pro_athlete.teams.office_holder",
            "temporal_field": "sports.pro_athlete.teams.from",
            "temporal_field_type": "date_range",
            "answer_type": "entity",
            "topic_label_hint": "athlete",
            "answer_label_hint": "team",
            "supported_templates": ["simple", "during", "first", "last"],
        },
        {
            "family": "marriage",
            "anchor_relation": "people.marriage.spouse",
            "answer_relation": "people.marriage.spouse.office_holder",
            "temporal_field": "people.marriage.spouse.from",
            "temporal_field_type": "date_range",
            "answer_type": "entity",
            "topic_label_hint": "person",
            "answer_label_hint": "spouse",
            "supported_templates": ["simple", "first", "last"],
        },
    ],
    "limits": {"max_facts_per_family": 200},
}


# ── SPARQL Miner Tests ─────────────────────────────────────────────────

class TestSparqlMinerUtils(unittest.TestCase):
    def test_normalize_mid_strips_prefixes(self):
        self.assertEqual(_normalize_mid("http://rdf.freebase.com/ns/m.02gjb"), "m.02gjb")
        self.assertEqual(_normalize_mid("fb:m.02gjb"), "m.02gjb")
        self.assertEqual(_normalize_mid("ns:m.02gjb"), "m.02gjb")
        self.assertEqual(_normalize_mid("m.02gjb"), "m.02gjb")

    def test_normalize_date_extracts_date(self):
        self.assertEqual(_normalize_date('"2005-12-31"^^xsd:dateTime'), "2005-12-31")
        self.assertEqual(_normalize_date("2005"), "2005")
        self.assertEqual(_normalize_date("2005-01"), "2005-01")

    def test_hash_row_deterministic(self):
        h1 = _hash_row("m.02gjb", "m.0jmc7", "2018", "sports.pro_athlete.teams")
        h2 = _hash_row("m.02gjb", "m.0jmc7", "2018", "sports.pro_athlete.teams")
        self.assertEqual(h1, h2)
        self.assertEqual(len(h1), 12)

        h3 = _hash_row("m.02gjb", "m.0jmc7", "2019", "sports.pro_athlete.teams")
        self.assertNotEqual(h1, h3)

    def test_summarize_mined_facts(self):
        facts = [
            _make_fact(fact_relation_family="sports_team"),
            _make_fact(fact_id="test-2", fact_relation_family="marriage",
                        topic_mid="m.xxx", answer_mid="m.yyy"),
        ]
        summary = summarize_mined_facts(facts)
        self.assertEqual(summary["total_facts"], 2)
        self.assertIn("sports_team", summary["by_family"])
        self.assertIn("marriage", summary["by_family"])
        self.assertEqual(summary["with_labels"], 2)
        self.assertEqual(summary["with_temporal_start"], 2)
        self.assertEqual(summary["unique_topics"], 2)
        self.assertEqual(len(summary["sample_facts"]), 2)


# ── Template Bank Tests ────────────────────────────────────────────────

class TestTemplateBank(unittest.TestCase):
    def test_template_bank_has_all_families(self):
        families = {t.family for t in TEMPLATES}
        # Current Zenodo-backed v1 pipeline only supports these four families.
        # before/after remain out of scope until real date literals are available.
        expected = {"simple", "during", "first", "last"}
        self.assertTrue(expected.issubset(families),
                        f"Missing: {expected - families}")

    def test_templates_have_valid_builders(self):
        valid = {"join", "tc", "argmin", "argmax"}
        for template in TEMPLATES:
            self.assertIn(template.s_expr_builder, valid,
                          f"{template.template_id}: unknown builder {template.s_expr_builder}")

    def test_generate_candidates_creates_samples(self):
        facts = [
            _make_fact(),
            _make_fact(fact_id="test-2",
                        fact_relation_family="marriage",
                        topic_mid="m.xxx", topic_label="John Doe",
                        answer_mid="m.yyy", answer_label="Jane Doe",
                        anchor_relation="people.marriage.spouse",
                        answer_relation="people.marriage.spouse.office_holder",
                        temporal_start="2000", temporal_end="2010",
                        retrieved_from_relation="people.marriage.spouse.from"),
        ]
        candidates = generate_candidates(facts, SAMPLE_WHITELIST, seed=42)
        self.assertGreater(len(candidates), 0)
        # All candidates should have valid S-expressions
        for c in candidates:
            self.assertIsNotNone(c.s_expression)
            self.assertNotEqual(c.s_expression, "")
            self.assertNotEqual(c.s_expression, "null")
            self.assertEqual(c.source, TemporalDataSource.SYNTHETIC)
            self.assertIn(c.metadata.get("template_family"),
                          ["simple", "during", "first", "last"])

    def test_generate_candidates_dedup(self):
        """Same fact + same template should not produce duplicates."""
        facts = [_make_fact()]
        candidates = generate_candidates(facts, SAMPLE_WHITELIST, seed=42)
        seen = set()
        for c in candidates:
            key = (c.question, c.s_expression)
            self.assertNotIn(key, seen, f"Duplicate: {key}")
            seen.add(key)

    def test_summarize_candidates(self):
        facts = [_make_fact()]
        candidates = generate_candidates(facts, SAMPLE_WHITELIST, seed=42)
        summary = summarize_candidates(candidates)
        self.assertGreater(summary["total_candidates"], 0)
        self.assertIn("by_template_family", summary)
        self.assertEqual(summary["total_candidates"], summary["unique_questions"])


# ── Verifier Tests ─────────────────────────────────────────────────────

class TestVerifierUtils(unittest.TestCase):
    def test_normalize_mid(self):
        self.assertEqual(normalize_mid("http://rdf.freebase.com/ns/m.02gjb"), "m.02gjb")
        self.assertEqual(normalize_mid("m.02gjb"), "m.02gjb")

    def test_is_mid(self):
        self.assertTrue(is_mid("m.02gjb"))
        self.assertTrue(is_mid("g.123abc"))
        self.assertFalse(is_mid("Barack Obama"))
        self.assertFalse(is_mid("2005"))

    def test_normalize_date(self):
        self.assertEqual(normalize_date("2010"), "2010")
        self.assertEqual(normalize_date("2010-01-01"), "2010-01-01")
        self.assertEqual(normalize_date("2010-1-1"), "2010-01-01")

    def test_is_date_like(self):
        self.assertTrue(is_date_like("2005"))
        self.assertTrue(is_date_like("2005-12"))
        self.assertTrue(is_date_like("2005-12-31"))
        self.assertFalse(is_date_like("m.02gjb"))
        self.assertFalse(is_date_like("hello"))

    def test_normalize_answer_set(self):
        result = normalize_answer_set(["m.02gjb", "http://rdf.freebase.com/ns/m.0jmc7", ""])
        self.assertIn("m.02gjb", result)
        self.assertIn("m.0jmc7", result)
        self.assertEqual(len(result), 2)

    def test_answers_match_mid_exact(self):
        self.assertTrue(answers_match(["m.02gjb"], ["m.02gjb"]))
        self.assertTrue(answers_match(
            ["http://rdf.freebase.com/ns/m.02gjb"],
            ["m.02gjb"]
        ))

    def test_answers_match_date_year(self):
        self.assertTrue(answers_match(["2005"], ["2005-01-01"]))
        self.assertTrue(answers_match(["2005-12-31"], ["2005"]))

    def test_answers_match_no_match(self):
        self.assertFalse(answers_match([], []))
        self.assertFalse(answers_match(["m.02gjb"], ["m.DIFFERENT"]))
        self.assertFalse(answers_match([], ["m.02gjb"]))

    def test_answers_match_label(self):
        # Labels normalized to lowercase should match
        self.assertTrue(answers_match(["Los Angeles Lakers"], ["los angeles lakers"]))
        self.assertTrue(answers_match(["  Barack Obama  "], ["barack obama"]))


# ── Distribution Filter Tests ──────────────────────────────────────────

class TestDistributionFilter(unittest.TestCase):
    def setUp(self):
        self.samples = []
        for i in range(20):
            family = "sports_team" if i < 10 else "marriage"
            samples_i = 10 if i < 10 else i - 10
            self.samples.append(_make_sample(
                sample_id=f"test-{i}",
                question=f"Who was the player of team {i}?",
                s_expression=f"(JOIN (R test.rel) m.{i:06d})",
                metadata={
                    "template_family": "during",
                    "fact_relation_family": family,
                    "temporal_start": f"{2000 + i * 2}",
                },
            ))

    def test_deduplicate_removes_dupes(self):
        duplicated = self.samples + [self.samples[0]]
        result = deduplicate(duplicated)
        self.assertEqual(len(result), len(self.samples))

    def test_cap_by_family(self):
        result = cap_by_family(self.samples, max_per_family=5, seed=42)
        by_fam = {}
        for s in result:
            fam = s.metadata["fact_relation_family"]
            by_fam[fam] = by_fam.get(fam, 0) + 1
        for fam, count in by_fam.items():
            self.assertLessEqual(count, 5, f"Family {fam} has {count} > 5")

    def test_cap_by_reasoning_type(self):
        result = cap_by_reasoning_type(self.samples, max_per_type=8, seed=42)
        self.assertLessEqual(len(result), 8)

    def test_apply_distribution_filter_pipeline(self):
        result = apply_distribution_filter(
            self.samples,
            max_per_family=5,
            max_per_type=10,
            max_per_date_bucket=8,
            seed=42,
        )
        self.assertLessEqual(len(result), len(self.samples))
        summary = summarize_distribution(result)
        self.assertIn("by_relation_family", summary)
        self.assertIn("by_date_bucket", summary)

    def test_filter_rejects_short_questions(self):
        short = _make_sample(sample_id="short", question="Hi", s_expression="(JOIN (R x) m.y)")
        result = apply_distribution_filter([short], min_sample_length=5, seed=42)
        self.assertEqual(len(result), 0)


# ── Training Export Tests ──────────────────────────────────────────────

class TestTrainingExport(unittest.TestCase):
    def test_sample_to_example(self):
        sample = _make_sample()
        example = sample_to_training_example(sample, include_metadata=True)
        self.assertIsNotNone(example)
        self.assertIn("instruction", example)
        self.assertIn("input", example)
        self.assertIn("output", example)
        self.assertIn("metadata", example)
        self.assertEqual(example["metadata"]["source"], "synthetic")

    def test_null_sexpr_skipped(self):
        sample = _make_sample(s_expression="null")
        example = sample_to_training_example(sample)
        self.assertIsNone(example)

    def test_empty_question_skipped(self):
        sample = _make_sample(question="   ")
        example = sample_to_training_example(sample)
        self.assertIsNone(example)

    def test_build_training_examples(self):
        samples = [_make_sample(sample_id=f"t-{i}",
                                question=f"Question {i}?",
                                s_expression=f"(JOIN (R test.rel) m.{i:06d})")
                   for i in range(10)]
        examples = build_temporal_training_examples(
            human_samples=[],
            synthetic_samples=samples,
            split="train",
            include_metadata=True,
        )
        self.assertEqual(len(examples), 10)
        for ex in examples:
            self.assertIn("instruction", ex)
            self.assertIn("output", ex)
            self.assertEqual(ex["history"], [])

    def test_exclude_suspicious_human(self):
        human = _make_sample(
            sample_id="human-1",
            source=TemporalDataSource.HUMAN,
            question="Who was president?",
            s_expression="(JOIN (R gov.pos) m.xxx)",
            metadata={"phase0_suspicious": True, "template_family": "simple"},
        )
        examples = build_temporal_training_examples(
            human_samples=[human],
            split="train",
            exclude_suspicious_human=True,
        )
        self.assertEqual(len(examples), 0)

        examples_allow = build_temporal_training_examples(
            human_samples=[human],
            split="train",
            exclude_suspicious_human=False,
        )
        self.assertEqual(len(examples_allow), 1)

    def test_dedup_across_human_and_synthetic(self):
        human = _make_sample(
            sample_id="human-1",
            source=TemporalDataSource.HUMAN,
            question="Q?",
            s_expression="(JOIN (R x) m.y)",
        )
        synth = _make_sample(
            sample_id="synth-1",
            source=TemporalDataSource.SYNTHETIC,
            question="Q?",  # same question
            s_expression="(JOIN (R x) m.y)",  # same output
        )
        examples = build_temporal_training_examples(
            human_samples=[human],
            synthetic_samples=[synth],
            split="train",
        )
        # Should deduplicate: only 1 example for the overlapping pair
        self.assertEqual(len(examples), 1)


# ── Schema Tests ───────────────────────────────────────────────────────

class TestSchema(unittest.TestCase):
    def test_temporal_sample_validation(self):
        valid = _make_sample()
        issues = valid.validate()
        self.assertEqual(issues, [])

    def test_temporal_sample_invalid(self):
        invalid = TemporalSample(
            sample_id="", question="", split="train",
            source=TemporalDataSource.SYNTHETIC,
            temporal_type=TemporalQuestionType.UNKNOWN,
            temporal_signal="", topic_entity_mid="",
            s_expression="null", sparql="", answers=[],
        )
        issues = invalid.validate()
        self.assertGreater(len(issues), 0)

    def test_sample_roundtrip(self):
        original = _make_sample()
        as_dict = original.to_dict()
        restored = TemporalSample.from_dict(as_dict)
        self.assertEqual(original.sample_id, restored.sample_id)
        self.assertEqual(original.question, restored.question)
        self.assertEqual(original.s_expression, restored.s_expression)
        self.assertEqual(original.source, restored.source)
        self.assertEqual(original.temporal_type, restored.temporal_type)


if __name__ == "__main__":
    unittest.main()
