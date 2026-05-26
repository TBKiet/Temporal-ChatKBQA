"""Tests for synthetic temporal sample quality filtering."""

import unittest

from src.temporal_data import (
    TemporalDataSource,
    TemporalQuestionType,
    TemporalSample,
    ValidationStatus,
    build_temporal_training_examples,
    filter_synthetic_samples,
    review_synthetic_sample,
    summarize_synthetic_filtering,
)


class TestTemporalQuality(unittest.TestCase):

    def _make_sample(
        self,
        sample_id: str = "synthetic-1",
        question: str = "Who held President during 2009-01-20 to 2017-01-20 for Barack Obama?",
        temporal_type: TemporalQuestionType = TemporalQuestionType.DURING,
        s_expression: str = "(TC (JOIN (R government.government_position_held) m.topic) government.government_position_held.from 2009-01-20)",
        metadata: dict | None = None,
    ) -> TemporalSample:
        return TemporalSample(
            sample_id=sample_id,
            question=question,
            split="train",
            source=TemporalDataSource.SYNTHETIC,
            temporal_type=temporal_type,
            temporal_signal="during" if temporal_type == TemporalQuestionType.DURING else temporal_type.value,
            topic_entity_mid="m.topic",
            s_expression=s_expression,
            sparql="",
            answers=["m.answer"],
            validation_status=ValidationStatus.RAW,
            source_dataset="SyntheticTemporal",
            relation_ids=["government.government_position_held.from"],
            entity_ids=["m.topic", "m.answer"],
            metadata=metadata or {
                "source_seed_id": "seed-1",
                "mined_fact_id": "fact-1",
            },
        )

    def test_review_synthetic_sample_accepts_valid_sample(self):
        sample = self._make_sample()
        issues = review_synthetic_sample(sample)
        self.assertEqual(issues, [])

    def test_review_synthetic_sample_rejects_unknown_placeholder(self):
        sample = self._make_sample(
            question="What happened during UNKNOWN for m.topic?",
            s_expression="(TC (JOIN (R government.government_position_held) m.topic) government.government_position_held.from UNKNOWN)",
        )
        issues = review_synthetic_sample(sample)
        self.assertIn("s_expression contains UNKNOWN placeholder", issues)
        self.assertIn("question still contains raw mids", issues)

    def test_filter_synthetic_samples_splits_accepted_and_rejected(self):
        accepted, rejected = filter_synthetic_samples(
            [
                self._make_sample(sample_id="ok-1"),
                self._make_sample(
                    sample_id="bad-1",
                    question="What was the last award for m.topic?",
                    temporal_type=TemporalQuestionType.LAST,
                    s_expression="(JOIN award.award_nominee.award_nominations m.topic)",
                ),
            ]
        )
        self.assertEqual(len(accepted), 1)
        self.assertEqual(len(rejected), 1)
        self.assertEqual(accepted[0].validation_status, ValidationStatus.NORMALIZED)
        self.assertEqual(rejected[0].validation_status, ValidationStatus.FAILED)
        self.assertTrue(rejected[0].metadata["filter_issues"])

    def test_summarize_synthetic_filtering(self):
        accepted, rejected = filter_synthetic_samples(
            [
                self._make_sample(sample_id="ok-1"),
                self._make_sample(
                    sample_id="bad-1",
                    question="What was the first award for m.topic?",
                    temporal_type=TemporalQuestionType.FIRST,
                    s_expression="(JOIN award.award_nominee.award_nominations m.topic)",
                ),
            ]
        )
        summary = summarize_synthetic_filtering(accepted, rejected)
        self.assertEqual(summary["accepted_samples"], 1)
        self.assertEqual(summary["rejected_samples"], 1)
        self.assertTrue(summary["top_rejection_reasons"])

    def test_training_export_can_require_filtered_synthetic(self):
        human_sample = TemporalSample(
            sample_id="human-1",
            question="Who was president in 1939?",
            split="train",
            source=TemporalDataSource.HUMAN,
            temporal_type=TemporalQuestionType.DURING,
            temporal_signal="during",
            topic_entity_mid="m.topic",
            s_expression="(TC (JOIN (R government.government_position_held) m.topic) government.government_position_held.from 1939)",
            sparql="SELECT ?x WHERE {}",
            answers=["m.answer"],
            validation_status=ValidationStatus.NORMALIZED,
            source_dataset="TempQuestions",
            relation_ids=["government.government_position_held.from"],
            entity_ids=["m.topic", "m.answer"],
            metadata={},
        )
        synthetic_samples = [
            self._make_sample(sample_id="ok-1"),
            self._make_sample(
                sample_id="bad-1",
                question="What was the last award for m.topic?",
                temporal_type=TemporalQuestionType.LAST,
                s_expression="(JOIN award.award_nominee.award_nominations m.topic)",
            ),
        ]
        examples = build_temporal_training_examples(
            human_samples=[human_sample],
            synthetic_samples=synthetic_samples,
            include_metadata=True,
            require_filtered_synthetic=True,
        )
        self.assertEqual(len(examples), 2)
        synthetic_count = sum(1 for example in examples if example["metadata"]["source"] == "synthetic")
        self.assertEqual(synthetic_count, 1)


if __name__ == "__main__":
    unittest.main()
