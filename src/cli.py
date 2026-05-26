"""Command-line interface for Temporal KBQA.

Usage:
    python -m src.cli --question "Who was US president before Obama?"
    python -m src.cli --question "..." --config configs/inference.yaml
    python -m src.cli --demo          # Run routing demo (no LLM required)
    python -m src.cli --interactive   # Interactive Q&A session
"""

import sys
import os
import argparse
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agent import detect_temporal_signals, demo_agent_routing


def _parse_args():
    parser = argparse.ArgumentParser(
        description='Temporal KBQA Command-Line Interface'
    )
    parser.add_argument('--question', type=str, default=None,
                        help='Question to answer')
    parser.add_argument('--config', type=str, default='configs/inference.yaml',
                        help='Path to inference config YAML')
    parser.add_argument('--mode', type=str, default=None,
                        choices=['temporal', 'standard'],
                        help='Force a specific pipeline mode (default: auto-detect)')
    parser.add_argument('--demo', action='store_true',
                        help='Run agent routing demo (no LLM required)')
    parser.add_argument('--interactive', action='store_true',
                        help='Start interactive Q&A session')
    parser.add_argument('--json', action='store_true',
                        help='Output result as JSON')
    return parser.parse_args()


def _print_result(result: dict, as_json: bool = False):
    if as_json:
        print(json.dumps(result, indent=2))
        return

    print("\n" + "=" * 60)
    print(f"Question : {result['question']}")
    print(f"Temporal : {result['is_temporal']} "
          f"(signals: {result['temporal_signals']})")
    print(f"Answer   : {result['answer']}")
    if result.get('temporal_constraint'):
        print(f"Constraint: {result['temporal_constraint']}")
    if result.get('sparql_used'):
        print(f"SPARQL   :\n{result['sparql_used']}")
    print(f"Retries  : {result['retries']}")
    print("\nReasoning steps:")
    for step in result['reasoning_steps']:
        print(f"  {step}")
    print("=" * 60)


def main():
    args = _parse_args()

    if args.demo:
        demo_agent_routing()
        return

    # Load pipeline + agent (requires Freebase + model)
    if not os.path.exists(args.config):
        print(f"Config not found: {args.config}")
        print("Run --demo for a routing demo that doesn't require the full setup.")
        sys.exit(1)

    from src.pipeline import TemporalKBQAPipeline
    from src.agent import TemporalQuestionAgent

    pipeline = TemporalKBQAPipeline.from_config(args.config)
    agent = TemporalQuestionAgent(pipeline)

    if args.interactive:
        print("Temporal KBQA Interactive Mode  (type 'quit' to exit)")
        print("-" * 60)
        while True:
            try:
                question = input("\nQuestion: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nExiting.")
                break
            if question.lower() in ('quit', 'exit', 'q'):
                break
            if not question:
                continue
            result = agent.run(question)
            _print_result(result, as_json=args.json)
        return

    if not args.question:
        print("Provide --question, --demo, or --interactive")
        sys.exit(1)

    result = agent.run(args.question)
    _print_result(result, as_json=args.json)


if __name__ == '__main__':
    main()
