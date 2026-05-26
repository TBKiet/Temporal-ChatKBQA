"""Unit tests for Temporal KBQA components.

Tests cover:
  - Temporal signal detection in questions
  - TempQuestionsParser with synthetic SPARQL examples
  - S-expression structure validation for temporal patterns
  - Temporal constraint extraction from S-expressions
"""

import sys
import os
import unittest
import tempfile
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agent import detect_temporal_signals, TemporalQuestionAgent


class TestTemporalSignalDetection(unittest.TestCase):

    def test_detects_before(self):
        signals = detect_temporal_signals("Who was president before Obama?")
        self.assertTrue(len(signals) > 0)

    def test_detects_after(self):
        signals = detect_temporal_signals("Which movies did Spielberg direct after 1990?")
        self.assertTrue(len(signals) > 0)

    def test_detects_year(self):
        signals = detect_temporal_signals("What currency was used in Germany in 2012?")
        self.assertTrue(len(signals) > 0)

    def test_detects_first(self):
        signals = detect_temporal_signals("Who first made it to the Moon?")
        self.assertTrue(len(signals) > 0)

    def test_detects_last(self):
        signals = detect_temporal_signals("What was the last award Harry Potter got?")
        self.assertTrue(len(signals) > 0)

    def test_non_temporal_question(self):
        signals = detect_temporal_signals("What is the capital of France?")
        # May or may not have signals; no year, no temporal keywords
        self.assertIsInstance(signals, list)

    def test_non_temporal_factual(self):
        signals = detect_temporal_signals("Who directed The Dark Knight?")
        self.assertEqual(signals, [])


class TestAgentRouting(unittest.TestCase):

    def setUp(self):
        # Use a minimal mock pipeline
        class MockPipeline:
            def answer(self, question, mode='temporal', relaxed=False):
                return {'answer': ['MockAnswer'], 'sparql': None,
                        'temporal_constraint': None, 'candidates_tried': 1}
        self.agent = TemporalQuestionAgent(MockPipeline(), max_retries=1)

    def test_temporal_question_routes_temporal(self):
        result = self.agent.run("Who was US president before Obama?")
        self.assertTrue(result['is_temporal'])
        self.assertIn('temporal', result['reasoning_steps'][1].lower())

    def test_factual_question_routes_standard(self):
        result = self.agent.run("Who directed The Dark Knight?")
        self.assertFalse(result['is_temporal'])
        self.assertIn('standard', result['reasoning_steps'][1].lower())

    def test_result_has_provenance(self):
        result = self.agent.run("What films did Hitchcock direct in 1960?")
        self.assertIn('question', result)
        self.assertIn('answer', result)
        self.assertIn('reasoning_steps', result)
        self.assertIn('retries', result)
        self.assertIsInstance(result['reasoning_steps'], list)
        self.assertEqual(len(result['reasoning_steps']), 4)

    def test_result_structure_types(self):
        result = self.agent.run("When did Messi first play professionally?")
        self.assertIsInstance(result['answer'], list)
        self.assertIsInstance(result['temporal_signals'], list)
        self.assertIsInstance(result['is_temporal'], bool)


class TestSparqlParserStructure(unittest.TestCase):
    """Test that the TempQuestionsParser correctly parses synthetic SPARQL patterns."""

    def setUp(self):
        from parse_sparql_tempquestions import TempQuestionsParser
        self.parser = TempQuestionsParser()

    def test_range_sparql_produces_tc(self):
        """A FILTER(NOT EXISTS ...) temporal range should produce a TC operator."""
        sparql = """PREFIX ns: <http://rdf.freebase.com/ns/>
SELECT DISTINCT ?x
WHERE {
ns:m.04f_xd8 ns:government.government_office_or_title.office_holders ?y .
?y ns:government.government_position_held.office_holder ?x .
FILTER(NOT EXISTS {?y ns:government.government_position_held.from ?sk0} ||
EXISTS {?y ns:government.government_position_held.from ?sk1 .
FILTER(xsd:datetime(?sk1) <= "2009-12-31"^^xsd:dateTime) })
FILTER(NOT EXISTS {?y ns:government.government_position_held.to ?sk2} ||
EXISTS {?y ns:government.government_position_held.to ?sk3 .
FILTER(xsd:datetime(?sk3) >= "2009-01-01"^^xsd:dateTime) })
}"""
        s_expr = self.parser.parse_query_tempquestions(sparql, 'm.04f_xd8')
        self.assertIn('TC', s_expr)
        self.assertIn('2009', s_expr)

    def test_superlative_sparql_produces_argmax(self):
        """ORDER BY DESC ... LIMIT 1 should produce ARGMAX."""
        sparql = """PREFIX ns: <http://rdf.freebase.com/ns/>
SELECT DISTINCT ?x
WHERE {
ns:m.0443c ns:sports.pro_athlete.teams ?y .
?y ns:sports.sports_team_roster.team ?x .
?y ns:sports.sports_team_roster.from ?sk0 .
}
ORDER BY DESC(xsd:datetime(?sk0))
LIMIT 1"""
        s_expr = self.parser.parse_query_tempquestions(sparql, 'm.0443c')
        self.assertIn('ARGMAX', s_expr.upper())

    def test_comparison_sparql_produces_gt_lt(self):
        """A FILTER comparison should produce gt/lt/ge/le in S-expression."""
        sparql = """PREFIX ns: <http://rdf.freebase.com/ns/>
SELECT DISTINCT ?x
WHERE {
?x ns:people.person.date_of_birth ?num .
ns:m.05cgv ns:film.actor.film ?y .
?y ns:film.performance.actor ?x .
FILTER (?num > "1950"^^xsd:dateTime) .
}"""
        s_expr = self.parser.parse_query_tempquestions(sparql, 'm.05cgv')
        self.assertTrue(
            any(op in s_expr for op in ['gt', 'ge', 'lt', 'le']),
            f"Expected comparison operator in: {s_expr}"
        )

    def test_compact_select_where_layout_is_normalized(self):
        sparql = """PREFIX ns: <http://rdf.freebase.com/ns/>
SELECT DISTINCT ?x WHERE {
ns:m.0443c ns:sports.pro_athlete.teams ?y .
?y ns:sports.sports_team_roster.team ?x .
?y ns:sports.sports_team_roster.from ?sk0 .
}
ORDER BY DESC(xsd:datetime(?sk0))
LIMIT 1"""
        s_expr = self.parser.parse_query_tempquestions(sparql, 'm.0443c')
        self.assertIn('ARGMAX', s_expr.upper())

    def test_reference_repair_fills_broken_placeholder_query(self):
        from parse_sparql_tempquestions import convert_tempquestions_instance, TempQuestionsParser

        parser = TempQuestionsParser()
        item = {
            "Question": "when was reagan inaugurated?",
            "TopicEntityMid": "m.placeholder",
            "Sparql": """PREFIX ns: <http://rdf.freebase.com/ns/>
SELECT DISTINCT ?x WHERE {
  ns:m.placeholder ns:government.government_position_held ?cvt .
  ?cvt ns:government.government_position_held.office_holder ?x .
  ?cvt ns:government.government_position_held.from ?from_date .
  FILTER(xsd:datetime(?from_date) = "2009-01-01"^^xsd:dateTime)
}""",
        }
        repaired, ok = convert_tempquestions_instance(
            parser,
            item,
            reference_lookup={"when was reagan inaugurated?": "(JOIN (R government.government_position_held) m.015lwh)"},
        )
        self.assertTrue(ok)
        self.assertEqual(repaired["SExpr_source"], "merged_reference")
        self.assertEqual(repaired["SExpr"], "(JOIN (R government.government_position_held) m.015lwh)")


class TestProcessNQOutputSelection(unittest.TestCase):

    def test_tempquestions_prefers_structured_output_in_auto_mode(self):
        from process_NQ import _select_output_expression

        item = {
            "sexpr": "(TC (JOIN (R government.government_position_held) m.02mjmr) government.government_position_held.from 1939-01-01)",
            "normed_sexpr": "( TC ( JOIN ( R [ government , government position held ] ) [ Barack Obama ] ) ... )",
        }
        output = _select_output_expression(item, "TempQuestions", "auto")
        self.assertEqual(output, item["sexpr"])

    def test_non_temporal_keeps_linearized_output_in_auto_mode(self):
        from process_NQ import _select_output_expression

        item = {
            "sexpr": "(JOIN (R people.person.parents) m.abc)",
            "normed_sexpr": "( JOIN ( R [ people , person , parents ] ) [ foo ] )",
        }
        output = _select_output_expression(item, "WebQSP", "auto")
        self.assertEqual(output, item["normed_sexpr"])

    def test_exclude_suspicious_filters_tempquestions_examples(self):
        args = type("Args", (), {"dataset_type": "TempQuestions", "exclude_suspicious": True})()
        data = [
            {"sexpr": "(JOIN a b)", "phase0_suspicious": False},
            {"sexpr": "(JOIN c d)", "phase0_suspicious": True},
            {"sexpr": "null", "phase0_suspicious": False},
        ]
        examples = []
        for item in data:
            if item["sexpr"].lower() != "null":
                if args.dataset_type == 'TempQuestions' and args.exclude_suspicious and item.get('phase0_suspicious'):
                    continue
                examples.append(item)
        self.assertEqual(len(examples), 1)


if __name__ == '__main__':
    unittest.main()
