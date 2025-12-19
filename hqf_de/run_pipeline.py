#!/usr/bin/env python3
import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def run_demo(text: str):
    from hqf_de.pipeline.expander import Expander

    print("\n" + "="*60)
    print("HQF-DE Demo")
    print("="*60)

    print(f"\nOriginal:\n{text[:500]}{'...' if len(text) > 500 else ''}")
    print("\nLoading...")

    exp = Expander()
    exp.load()

    print("Expanding...")
    result = exp.expand("demo", text)

    print("\n" + "-"*40)
    print("RESULTS")
    print("-"*40)

    print("\nGaps:")
    for i, g in enumerate(result.gaps, 1):
        print(f"   {i}. {g}")

    print("\nValid:")
    for e in result.valid:
        print(f"   - {e}")

    if result.rejected:
        print("\nRejected:")
        for e in result.rejected:
            print(f"   - {e}")

    print("\nQueries:")
    for q in result.queries:
        print(f"   - {q}")

    print("\n" + "-"*40)
    print("EXPANDED")
    print("-"*40)
    print(result.expanded)

    print(f"\nStats: {len(result.final)} expansions")
    exp.unload()


def run_expansion(limit: int = None, d2q_only: bool = False):
    from hqf_de.pipeline.expander import Expander
    from hqf_de.pipeline.indexer_bridge import Bridge
    from hqf_de.config import config

    print("\n" + "="*60)
    print("HQF-DE Expansion")
    print("="*60)

    bridge = Bridge()

    input_path = config.data_dir / config.input_tsv
    if not input_path.exists():
        test_path = config.data_dir.parent / "test.tsv"
        if test_path.exists():
            input_path = test_path
            print(f"Using: {input_path}")
        else:
            print(f"\nNot found: {input_path}")
            return

    docs = list(bridge.read(str(input_path.name), limit=limit))
    print(f"\nLoaded {len(docs)} docs")

    if d2q_only:
        print("\nMode: d2q only")
        exp = Expander(use_llm=False, use_nli=False, use_d2q=True)
    else:
        print("\nMode: full HQF-DE")
        exp = Expander()

    print("\nLoading...")
    exp.load()

    print("Expanding...")
    results = []
    for i, (doc_id, text) in enumerate(docs):
        if d2q_only:
            r = exp.d2q_only(doc_id, text)
        else:
            r = exp.expand(doc_id, text)
        results.append(r)
        if (i + 1) % 10 == 0:
            print(f"   {i + 1}/{len(docs)}")

    output_name = "expanded_d2q.tsv" if d2q_only else "expanded_hqfde.tsv"
    n, path = bridge.write(iter(results), filename=output_name)

    print(f"\nDone! {n} docs -> {path}")
    avg = sum(len(r.final) for r in results) / len(results)
    print(f"Avg expansions: {avg:.1f}")

    exp.unload()


def run_eval(num_queries: int = 100, num_docs: int = 1000):
    from hqf_de.pipeline.expander import Expander
    from hqf_de.pipeline.indexer_bridge import Bridge
    from hqf_de.evaluation.evaluator import Evaluator

    print("\n" + "="*60)
    print("HQF-DE Evaluation")
    print("="*60)

    print(f"\nDocs: {num_docs}, Queries: {num_queries}")

    bridge = Bridge()
    docs = list(bridge.read(limit=num_docs))

    if not docs:
        print("\nNo docs found")
        return

    print(f"\nLoaded {len(docs)} docs")
    print("Expanding...")

    exp = Expander()
    with exp:
        results = []
        for i, (doc_id, text) in enumerate(docs):
            r = exp.expand(doc_id, text)
            results.append(r)
            if (i + 1) % 50 == 0:
                print(f"   {i + 1}/{len(docs)}")

    print("\nEvaluating...")

    ev = Evaluator()
    eval_results = ev.compare(results, num_queries=num_queries)

    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)

    rep = ev.report(eval_results)
    print(rep)

    ev.save(eval_results)
    print(f"\nSaved: {ev.output_dir / 'eval.json'}")


def main():
    parser = argparse.ArgumentParser(description="HQF-DE")

    parser.add_argument("--demo", type=str, help="Demo text")
    parser.add_argument("--expand", action="store_true", help="Expand docs")
    parser.add_argument("--d2q-only", action="store_true", help="D2Q only")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate")
    parser.add_argument("--limit", type=int, default=None, help="Doc limit")
    parser.add_argument("--queries", type=int, default=100, help="Query count")

    args = parser.parse_args()

    if args.demo:
        run_demo(args.demo)
    elif args.expand:
        run_expansion(limit=args.limit, d2q_only=args.d2q_only)
    elif args.evaluate:
        run_eval(num_queries=args.queries, num_docs=args.limit or 1000)
    else:
        parser.print_help()
        print("\nExamples:")
        print('  python run_pipeline.py --demo "The quick brown fox."')
        print("  python run_pipeline.py --expand --limit 100")
        print("  python run_pipeline.py --expand --d2q-only --limit 100")
        print("  python run_pipeline.py --evaluate --queries 50 --limit 500")


if __name__ == "__main__":
    main()
