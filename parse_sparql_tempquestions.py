"""Parse TempQuestions SPARQL queries into S-expressions with temporal operators.

TempQuestions uses Freebase CVTs for temporal facts (marriage.date, position.start_date, etc.).
We reuse the WebQSP Parser class since both datasets use Freebase with the same SPARQL patterns.
The key addition is handling TempQuestions-specific temporal SPARQL formats.

TempQuestions SPARQL patterns handled:
  - CVT-based temporal range:  FILTER(NOT EXISTS {?y ns:rel.from ...} || EXISTS {...})
  - Temporal comparison:       FILTER(?num > "2009"^^xsd:dateTime)
  - Ordinal/superlative:       ORDER BY DESC(?sk0) LIMIT 1
  - Simple temporal join:      ?x ns:event.start_date ?sk0 . FILTER(?sk0 = "2012"^^xsd:dateTime)
"""

import re
import os
import json
from tqdm import tqdm
from components.utils import load_json, dump_json
from executor.logic_form_util import lisp_to_sparql
from parse_sparql_webqsp import Parser


class TempQuestionsParser(Parser):
    """Extends WebQSP Parser with TempQuestions-specific SPARQL patterns."""

    @staticmethod
    def _canonicalize_query_layout(query: str) -> str:
        """Normalize compact TempQuestions SPARQL into a parser-friendly layout."""
        query = query.replace("\r\n", "\n").replace("\r", "\n").strip()
        query = re.sub(r"SELECT DISTINCT \?x\s+WHERE\s*\{", "SELECT DISTINCT ?x\nWHERE {", query)
        query = re.sub(r"SELECT \?x\s+WHERE\s*\{", "SELECT ?x\nWHERE {", query)
        query = re.sub(r"\}\s*ORDER BY", "}\nORDER BY", query)
        query = re.sub(r"ORDER BY ([^\n]+)\s+LIMIT 1", r"ORDER BY \1\nLIMIT 1", query)
        return query

    def parse_query_tempquestions(self, query: str, topic_mid: str) -> str:
        """Parse a TempQuestions SPARQL query into an S-expression.

        Handles TempQuestions-specific patterns beyond standard WebQSP:
        - Implicit temporal questions using CVT date attributes
        - Explicit year/date filter patterns
        - Before/After signal patterns encoded as FILTER comparisons
        """
        query = self._canonicalize_query_layout(query)
        lines = query.strip().split('\n')
        lines = [x.strip() for x in lines if x.strip()]

        if not lines:
            raise ValueError("Empty query")

        # Skip manual SPARQL markers
        if lines[0] == '#MANUAL SPARQL':
            raise AssertionError("Manual SPARQL not supported")

        # Skip PREFIX lines
        line_num = 0
        while line_num < len(lines) and lines[line_num].startswith('PREFIX'):
            line_num += 1

        # Validate SELECT line
        if not lines[line_num].startswith('SELECT DISTINCT ?x') and \
           not lines[line_num].startswith('SELECT ?x'):
            raise AssertionError(f"Unexpected SELECT line: {lines[line_num]}")
        line_num += 1

        if lines[line_num] != 'WHERE {':
            raise AssertionError(f"Expected WHERE {{, got: {lines[line_num]}")

        # Normalize ORDER BY LIMIT on same line
        if re.match(r'ORDER BY .*\?\w*.* LIMIT 1', lines[-1]):
            lines[-1] = lines[-1].replace('LIMIT 1', '').strip()
            lines.append('LIMIT 1')

        if re.match(r'LIMIT \d+', lines[-1]):
            lines[-1] = 'LIMIT 1'

        if lines[-1].startswith('OFFSET'):
            lines.pop(-1)

        if lines[-1] not in ['}', 'LIMIT 1']:
            raise AssertionError(f"Unexpected last line: {lines[-1]}")

        lines = lines[line_num:]

        filter_string_flag = any('FILTER (str' in x for x in lines)

        body_lines, spec_condition, filter_lines = self.normalize_body_lines(
            lines, filter_string_flag)
        body_lines = [x.strip() for x in body_lines]

        # Handle predefined filter pairs at start of body
        if body_lines and body_lines[0].startswith('FILTER'):
            predefined_filter0 = body_lines[0]
            predefined_filter1 = body_lines[1] if len(body_lines) > 1 else ''

            filter_0_valid = (predefined_filter0 == 'FILTER (?x != ?c)')
            if not filter_0_valid and topic_mid:
                filter_0_valid = (predefined_filter0 == f'FILTER (?x != ns:{topic_mid})')
                filter_0_valid = filter_0_valid or (predefined_filter0 == f'FILTER (?x != {topic_mid})')

            if filter_0_valid and "isLiteral" in predefined_filter1:
                body_lines = body_lines[2:]

        # Validate body lines
        valid_starts = ('?', 'ns:')
        if not all(x.startswith(valid_starts) for x in body_lines):
            invalid = [x for x in body_lines if not x.startswith(valid_starts)]
            raise AssertionError(f"Invalid body lines: {invalid}")

        mid_list = [f'ns:{topic_mid}'] if topic_mid else []
        var_dep_list = self.parse_naive_body(body_lines, filter_lines, '?x', spec_condition)
        s_expr = self.dep_graph_to_s_expr(var_dep_list, '?x', spec_condition)
        return s_expr


def _load_reference_sexpr_lookup(split: str) -> dict[str, str]:
    """Load a trusted question -> sexpr lookup from merged artifacts if present.

    The raw TempQuestions files bundled in this repo currently contain
    placeholder-corrupted SPARQL for many items. Until a clean upstream export is
    restored, we use the aligned merged artifacts as a repair source so the
    pipeline remains reproducible.
    """
    merged_path = f"data/TempQuestions/generation/merged/TempQuestions_{split}.json"
    if not os.path.exists(merged_path):
        return {}

    lookup: dict[str, str] = {}
    for item in load_json(merged_path):
        question = (item.get("question") or "").strip().lower()
        sexpr = item.get("sexpr") or item.get("normed_sexpr") or "null"
        if question and sexpr and str(sexpr).lower() != "null":
            lookup[question] = sexpr
    return lookup


def _repair_with_reference(item: dict, reference_lookup: dict[str, str]) -> tuple[dict, bool]:
    question = (item.get("Question") or item.get("question") or "").strip().lower()
    if question and question in reference_lookup:
        item["SExpr"] = reference_lookup[question]
        item["SExpr_source"] = "merged_reference"
        return item, True
    item["SExpr"] = "null"
    item["SExpr_source"] = "parser_failed"
    return item, False


def convert_tempquestions_instance(
    parser: TempQuestionsParser,
    item: dict,
    reference_lookup: dict[str, str] | None = None,
) -> dict:
    """Convert a single TempQuestions item to include S-expression."""
    sparql = item.get('Sparql', item.get('sparql', ''))
    topic_mid = item.get('TopicEntityMid', item.get('topic_mid', ''))
    reference_lookup = reference_lookup or {}

    try:
        s_expr = parser.parse_query_tempquestions(sparql, topic_mid)
        item['SExpr'] = s_expr
        item['SExpr_source'] = 'parser'
        return item, s_expr != 'null'
    except (AssertionError, ValueError, KeyError, IndexError):
        return _repair_with_reference(item, reference_lookup)


def augment_tempquestions_with_sexpr(split: str, check_execute_accuracy: bool = False):
    """Process TempQuestions split and augment with S-expressions."""
    input_file = f'data/TempQuestions/origin/TempQuestions.{split}.json'
    if not os.path.exists(input_file):
        print(f"Input file not found: {input_file}")
        print("Please download TempQuestions dataset and place it in data/TempQuestions/origin/")
        return

    dataset = load_json(input_file)
    parser = TempQuestionsParser()
    reference_lookup = _load_reference_sexpr_lookup(split)

    total_num = 0
    hit_num = 0
    execute_hit_num = 0
    source_counter = {"parser": 0, "merged_reference": 0, "parser_failed": 0}

    for i, item in tqdm(enumerate(dataset), total=len(dataset)):
        total_num += 1
        item, flag_success = convert_tempquestions_instance(
            parser,
            item,
            reference_lookup=reference_lookup,
        )
        source_counter[item.get("SExpr_source", "parser_failed")] += 1

        if flag_success:
            hit_num += 1
            if check_execute_accuracy:
                execute_right_flag = False
                try:
                    from executor.sparql_executor import execute_query_with_odbc
                    sparql_query = lisp_to_sparql(item['SExpr'])
                    execute_ans = execute_query_with_odbc(sparql_query)
                    execute_ans = {
                        r.replace('http://rdf.freebase.com/ns/', '') for r in execute_ans
                    }
                    gold_ans = set(item.get('Answers', []))
                    execute_right_flag = (execute_ans == gold_ans or
                                          bool(execute_ans & gold_ans))
                    item['SExpr_execute_right'] = execute_right_flag
                    if execute_right_flag:
                        execute_hit_num += 1
                except Exception:
                    item['SExpr_execute_right'] = False

        if (i + 1) % 100 == 0:
            print(f'[{split}] Processed {i+1}: S-Expr rate {hit_num}/{total_num} '
                  f'({hit_num/total_num:.2%})')

    print(f'[{split}] Final S-Expr rate: {hit_num}/{total_num} ({hit_num/total_num:.2%})')
    print(f'[{split}] Sources: {source_counter}')
    if check_execute_accuracy:
        print(f'[{split}] Execute accuracy: {execute_hit_num}/{total_num} '
              f'({execute_hit_num/total_num:.2%})')

    output_dir = 'data/TempQuestions/sexpr'
    os.makedirs(output_dir, exist_ok=True)
    output_file = f'{output_dir}/TempQuestions.{split}.expr.json'
    print(f'Writing S-expressions to {output_file}')
    dump_json(dataset, output_file, indent=4)


def parse_tempquestions_sparql(check_execute_accuracy: bool = False):
    """Entry point: process all TempQuestions splits."""
    for split in ['train', 'test']:
        augment_tempquestions_with_sexpr(split, check_execute_accuracy)


if __name__ == '__main__':
    import argparse
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--check_execute', action='store_true',
                            help='Verify S-expression execution accuracy against Freebase')
    args = arg_parser.parse_args()
    parse_tempquestions_sparql(check_execute_accuracy=args.check_execute)
