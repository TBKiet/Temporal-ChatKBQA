"""Tests for canonical temporal dataset utilities."""

import unittest

from src.temporal_data import (
    TemporalDataSource,
    TemporalQuestionType,
    TemporalSample,
    ValidationStatus,
    build_temporal_relation_inventory,
    infer_temporal_question_type,
    standardize_tempquestions_split,
    summarize_temporal_samples,
)


class TestTemporalDatasetSchema(unittest.TestCase):

    def test_infer_before_question(self):
        t = infer_temporal_question_type(
            question="Who was president before Obama?",
            temporal_signal="before",
        )
        self.assertEqual(t, TemporalQuestionType.BEFORE)

    def test_infer_last_question(self):
        t = infer_temporal_question_type(
            question="What was the last award Harry Potter got?",
            s_expression="(ARGMAX (JOIN ...) award.award_nomination.award_nomination_year)",
        )
        self.assertEqual(t, TemporalQuestionType.LAST)

    def test_infer_explicit_date_question(self):
        t = infer_temporal_question_type(
            question="What movies did Hitchcock direct in 1960?",
            temporal_signal="during",
        )
        self.assertEqual(t, TemporalQuestionType.DURING)

    def test_sample_validation_passes(self):
        sample = TemporalSample(
            sample_id="tmp-1",
            question="Who was president before Obama?",
            split="train",
            source=TemporalDataSource.HUMAN,
            temporal_type=TemporalQuestionType.BEFORE,
            temporal_signal="before",
            topic_entity_mid="m.02mjmr",
            s_expression="(TC (JOIN ...) government.government_position_held.from 2009-01-01)",
            sparql="SELECT ?x WHERE { ?x ?p ?o }",
            answers=["George W. Bush"],
            validation_status=ValidationStatus.NORMALIZED,
        )
        self.assertEqual(sample.validate(), [])

    def test_sample_validation_flags_unknown_type(self):
        sample = TemporalSample(
            sample_id="tmp-2",
            question="Temporal question",
            split="train",
            source=TemporalDataSource.SYNTHETIC,
            temporal_type=TemporalQuestionType.UNKNOWN,
            temporal_signal="during",
            topic_entity_mid="",
            s_expression="(JOIN foo bar)",
            sparql="SELECT ?x WHERE { ?x ?p ?o }",
            answers=["answer"],
        )
        self.assertIn("temporal_type should be resolved before export", sample.validate())


class TestTemporalRelationInventory(unittest.TestCase):

    def test_inventory_counts_temporal_relations(self):
        samples = [
            TemporalSample(
                sample_id="tmp-3",
                question="Who was president before Obama?",
                split="train",
                source=TemporalDataSource.HUMAN,
                temporal_type=TemporalQuestionType.BEFORE,
                temporal_signal="before",
                topic_entity_mid="m.02mjmr",
                s_expression="(TC (JOIN (R government.government_position_held) m.02mjmr) government.government_position_held.from 2009-01-01)",
                sparql="SELECT ?x WHERE { ?x ?p ?o }",
                answers=["George W. Bush"],
                relation_ids=["government.government_position_held.from"],
            )
        ]
        inventory = build_temporal_relation_inventory(samples, ontology_path=None)
        self.assertTrue(inventory)
        self.assertEqual(inventory[0].relation_id, "government.government_position_held.from")

    def test_inventory_avoids_common_topic_false_positive(self):
        samples = [
            TemporalSample(
                sample_id="tmp-6",
                question="Who was president before Obama?",
                split="train",
                source=TemporalDataSource.HUMAN,
                temporal_type=TemporalQuestionType.BEFORE,
                temporal_signal="before",
                topic_entity_mid="m.02mjmr",
                s_expression="(JOIN common.topic m.02mjmr)",
                sparql="SELECT ?x WHERE { ?x ?p ?o }",
                answers=["George W. Bush"],
                relation_ids=["common.topic"],
            )
        ]
        inventory = build_temporal_relation_inventory(samples, ontology_path=None)
        self.assertEqual(inventory, [])

    def test_summary_reports_counts(self):
        samples = [
            TemporalSample(
                sample_id="tmp-4",
                question="Who was president before Obama?",
                split="train",
                source=TemporalDataSource.HUMAN,
                temporal_type=TemporalQuestionType.BEFORE,
                temporal_signal="before",
                topic_entity_mid="m.02mjmr",
                s_expression="(TC (JOIN ...) government.government_position_held.from 2009-01-01)",
                sparql="SELECT ?x WHERE { ?x ?p ?o }",
                answers=["George W. Bush"],
            ),
            TemporalSample(
                sample_id="tmp-5",
                question="What was the last award Harry Potter got?",
                split="train",
                source=TemporalDataSource.SYNTHETIC,
                temporal_type=TemporalQuestionType.LAST,
                temporal_signal="last",
                topic_entity_mid="m.02q4m",
                s_expression="(ARGMAX (JOIN ...) award.award_nomination.award_nomination_year)",
                sparql="SELECT ?x WHERE { ?x ?p ?o }",
                answers=["Nestle Children's Book Prize"],
            ),
        ]
        summary = summarize_temporal_samples(samples)
        self.assertEqual(summary["total_samples"], 2)
        self.assertEqual(summary["by_temporal_type"]["before"], 1)
        self.assertEqual(summary["by_temporal_type"]["last"], 1)


class TestTempQuestionsStandardization(unittest.TestCase):

    def test_standardize_prefers_audited_merged_when_available(self):
        import json
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            merged = tmp / "TempQuestions_train.json"
            audited = tmp / "TempQuestions_train.audited.json"
            origin = tmp / "TempQuestions.train.json"

            merged.write_text(json.dumps([{
                "ID": "1",
                "question": "when was 13 going on 30 released?",
                "answer": ["2004-04-14"],
                "sexpr": "(JOIN (R government.government_position_held) m.bad)",
                "normed_sexpr": "(JOIN (R government.government_position_held) m.bad)",
                "relations": ["government.government_position_held"],
                "entities": ["m.bad"],
            }]), encoding="utf-8")
            audited.write_text(json.dumps([{
                "ID": "1",
                "question": "when was 13 going on 30 released?",
                "answer": ["2004-04-14"],
                "sexpr": "(JOIN (R film.film.initial_release_date) m.good)",
                "normed_sexpr": "(JOIN (R film.film.initial_release_date) m.good)",
                "relations": ["film.film.initial_release_date"],
                "entities": ["m.good"],
                "phase0_suspicious": True,
                "phase0_suspicious_reasons": ["generic_government_relation_for_non_government_question"],
                "phase0_inferred_temporal_type": "temporal_answer",
                "phase0_origin_question_type": "Temp.Ans",
            }]), encoding="utf-8")
            origin.write_text(json.dumps([{
                "Id": 1,
                "Question": "when was 13 going on 30 released?",
                "TemporalSignalNorm": "NONE",
                "Temporal signal": ["No Signal"],
                "Type": ["Temp.Ans"],
                "Data source": "Free917",
                "Question creation date": "x",
                "TopicEntityMid": "m.placeholder",
                "Sparql": "SELECT DISTINCT ?x WHERE {}",
            }]), encoding="utf-8")

            samples = standardize_tempquestions_split(merged, origin, split="train")
            self.assertEqual(samples[0].s_expression, "(JOIN (R film.film.initial_release_date) m.good)")
            self.assertTrue(samples[0].metadata["phase0_suspicious"])


if __name__ == "__main__":
    unittest.main()
