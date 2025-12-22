#!/usr/bin/env python3
import argparse

def run_demo(text):
    from hqf_de.pipeline.expander import Expander

    print(f"\n{'='*60}\nHQF-DE Demo\n{'='*60}")
    print(f"\nOriginal: {text[:500]}{'...' if len(text) > 500 else ''}")

    exp = Expander()
    exp.load()
    result = exp.expand("demo", text)

    print(f"\nGaps: {result.get('gaps', [])}")
    print(f"Expansions: {result.get('final', [])}")
    print(f"\nExpanded: {result.get('expanded', text)}")

    exp.unload()


def run_expansion(limit=None, d2q_only=False):
    from hqf_de.pipeline.expander import Expander
    from hqf_de.pipeline.indexer_bridge import Bridge
    from hqf_de.config import config

    print(f"\n{'='*60}\nHQF-DE Expansion\n{'='*60}")

    bridge = Bridge()
    input_path = config.data_dir / config.input_tsv
    if not input_path.exists():
        print(f"Not found: {input_path}")
        return

    docs = list(bridge.read(str(input_path.name), limit=limit))
    print(f"\n{len(docs)} docs loaded")

    exp = Expander(use_llm=not d2q_only, use_nli=not d2q_only, use_d2q=True)
    exp.load()

    results = []
    for i, (doc_id, text) in enumerate(docs):
        r = exp.d2q_only(doc_id, text) if d2q_only else exp.expand(doc_id, text)
        results.append(r)
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(docs)}")

    output_name = "expanded_d2q.tsv" if d2q_only else "expanded_hqfde.tsv"
    n, path = bridge.write(iter(results), filename=output_name)
    print(f"\nDone! {n} docs -> {path}")

    exp.unload()


def run_eval(num_queries=100, num_docs=1000):
    from hqf_de.pipeline.expander import Expander
    from hqf_de.pipeline.indexer_bridge import Bridge
    from hqf_de.evaluation.evaluator import Evaluator

    print(f"\n{'='*60}\nHQF-DE Evaluation\n{'='*60}")
    print(f"Docs: {num_docs}, Queries: {num_queries}")

    bridge = Bridge()
    docs = list(bridge.read(limit=num_docs))
    if not docs:
        print("No docs found")
        return

    exp = Expander()
    with exp:
        results = [exp.expand(doc_id, text) for doc_id, text in docs]

    ev = Evaluator()
    eval_results = ev.compare(results, num_queries=num_queries)
    print(ev.report(eval_results))
    ev.save(eval_results)


def main():
    parser = argparse.ArgumentParser(description="HQF-DE Pipeline")
    parser.add_argument("--demo", type=str)
    parser.add_argument("--expand", action="store_true")
    parser.add_argument("--d2q-only", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--queries", type=int, default=100)

    args = parser.parse_args()

    if args.demo:
        run_demo(args.demo)
    elif args.expand:
        run_expansion(limit=args.limit, d2q_only=args.d2q_only)
    elif args.evaluate:
        run_eval(num_queries=args.queries, num_docs=args.limit or 1000)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
