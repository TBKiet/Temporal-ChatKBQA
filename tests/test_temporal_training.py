"""Tests for temporal training-data export."""

import unittest

from src import STANDARD_INSTRUCTION, TEMPORAL_INSTRUCTION
from src.temporal_data import (
    TemporalDataSource,
    TemporalQuestionType,
    TemporalSample,
    ValidationStatus,
    build_temporal_training_examples,
    sample_to_training_example,
    summarize_training_examples,
)


class TestTemporalTraining(unittest.TestCase):

    def _make_sample(
        self,
        sample_id: str,
        source: TemporalDataSource,
        temporal_type: TemporalQuestionType,
        split: str = "train",
        question: str = "Who was president in 1939?",
        s_expression: str = "(TC (JOIN (R government.government_position_held) m.topic) government.government_position_held.from 1939)",
    ) -> TemporalSample:
        return TemporalSample(
            sample_id=sample_id,
            question=question,
            split=split,
            source=source,
            temporal_type=temporal_type,
            temporal_signal="during",
            topic_entity_mid="m.topic",
            s_expression=s_expression,
            sparql="SELECT ?x WHERE {}",
            answers=["m.answer"],
            validation_status=ValidationStatus.NORMALIZED,
            source_dataset="TempQuestions" if source == TemporalDataSource.HUMAN else "SyntheticTemporal",
        )

    def test_sample_to_training_example_uses_temporal_instruction(self):
        sample = self._make_sample("human-1", TemporalDataSource.HUMAN, TemporalQuestionType.DURING)
        example = sample_to_training_example(sample, include_metadata=True)
        self.assertIsNotNone(example)
        self.assertEqual(example["instruction"], TEMPORAL_INSTRUCTION)
        self.assertEqual(example["metadata"]["source"], "human")

    def test_sample_to_training_example_uses_standard_instruction_for_unknown(self):
        sample = self._make_sample("human-2", TemporalDataSource.HUMAN, TemporalQuestionType.UNKNOWN)
        example = sample_to_training_example(sample)
        self.assertEqual(example["instruction"], STANDARD_INSTRUCTION)

    def test_build_temporal_training_examples_combines_human_and_synthetic(self):
        human_samples = [
            self._make_sample("human-1", TemporalDataSource.HUMAN, TemporalQuestionType.DURING),
        ]
        synthetic_samples = [
            self._make_sample(
                "synthetic-1",
                TemporalDataSource.SYNTHETIC,
                TemporalQuestionType.LAST,
                question="What was the last award for Harry Potter?",
                s_expression="(ARGMAX (JOIN (R award.award_nominee.award_nominations) m.topic) award.award_nomination.award_nomination_year)",
            ),
        ]

        examples = build_temporal_training_examples(
            human_samples=human_samples,
            synthetic_samples=synthetic_samples,
            include_metadata=True,
        )
        self.assertEqual(len(examples), 2)
        self.assertEqual(examples[0]["metadata"]["source"], "human")
        self.assertEqual(examples[1]["metadata"]["source"], "synthetic")

    def test_build_temporal_training_examples_deduplicates(self):
        sample = self._make_sample("human-1", TemporalDataSource.HUMAN, TemporalQuestionType.DURING)
        duplicate = self._make_sample("human-2", TemporalDataSource.HUMAN, TemporalQuestionType.DURING)
        examples = build_temporal_training_examples([sample, duplicate], include_metadata=True)
        self.assertEqual(len(examples), 1)

    def test_build_temporal_training_examples_respects_synthetic_cap(self):
        human_samples = [
            self._make_sample("human-1", TemporalDataSource.HUMAN, TemporalQuestionType.DURING),
        ]
        synthetic_samples = [
            self._make_sample(
                "synthetic-1",
                TemporalDataSource.SYNTHETIC,
                TemporalQuestionType.LAST,
                question="What was the last award for Harry Potter?",
                s_expression="(ARGMAX (JOIN (R award.award_nominee.award_nominations) m.topic) award.award_nomination.award_nomination_year)",
            ),
            self._make_sample(
                "synthetic-2",
                TemporalDataSource.SYNTHETIC,
                TemporalQuestionType.FIRST,
                question="What was the first award for Harry Potter?",
                s_expression="(ARGMIN (JOIN (R award.award_nominee.award_nominations) m.topic) award.award_nomination.award_nomination_year)",
            ),
        ]
        examples = build_temporal_training_examples(
            human_samples=human_samples,
            synthetic_samples=synthetic_samples,
            max_synthetic=1,
            include_metadata=True,
        )
        self.assertEqual(len(examples), 2)
        synthetic_count = sum(1 for example in examples if example["metadata"]["source"] == "synthetic")
        self.assertEqual(synthetic_count, 1)

    def test_summarize_training_examples(self):
        human_samples = [
            self._make_sample("human-1", TemporalDataSource.HUMAN, TemporalQuestionType.DURING),
        ]
        examples = build_temporal_training_examples(human_samples, include_metadata=True)
        summary = summarize_training_examples(examples)
        self.assertEqual(summary["total_examples"], 1)
        self.assertEqual(summary["by_source"]["human"], 1)

    def test_build_temporal_training_examples_can_exclude_suspicious_human(self):
        suspicious_human = self._make_sample("human-1", TemporalDataSource.HUMAN, TemporalQuestionType.DURING)
        suspicious_human.metadata["phase0_suspicious"] = True
        clean_human = self._make_sample("human-2", TemporalDataSource.HUMAN, TemporalQuestionType.DURING)
        examples = build_temporal_training_examples(
            human_samples=[suspicious_human, clean_human],
            include_metadata=True,
            exclude_suspicious_human=True,
        )
        self.assertEqual(len(examples), 1)
        self.assertEqual(examples[0]["metadata"]["sample_id"], "human-2")


if __name__ == "__main__":
    unittest.main()
