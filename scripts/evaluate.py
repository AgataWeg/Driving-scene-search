"""Evaluate retrieval quality and compare models (CLIP vs. SigLIP).

Uses the prebuilt indexes in indexes/<model>; only text queries are embedded.
"""

import argparse
from pathlib import Path

import pandas as pd

from src.embed import get_embedder
from src.evaluate import evaluate
from src.search import Searcher
from src.rewrite import rewrite_query
from dotenv import load_dotenv


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models", nargs="+", choices=["clip", "siglip"], default=["clip", "siglip"]
    )
    parser.add_argument("--frames", default="data/frames.csv")
    parser.add_argument("--ks", nargs="+", type=int, default=[5, 10, 50])
    parser.add_argument("--out")
    parser.add_argument(
        "--rewrite",
        action="store_true",
        help="rewrite the query with an LLM before searching",
    )
    args = parser.parse_args()
    load_dotenv()
    out = args.out or (
        "results/eval_rewrite.csv" if args.rewrite else "results/eval.csv"
    )

    results = []
    for model in args.models:
        searcher = Searcher(get_embedder(model), f"indexes/{model}", args.frames)
        df = evaluate(
            searcher, ks=tuple(args.ks), rewrite=rewrite_query if args.rewrite else None
        )
        df.insert(0, "model", model)
        results.append(df)
    results = pd.concat(results, ignore_index=True)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_colwidth", 45)
    metric_cols = [
        c
        for c in results.columns
        if c.startswith(("P@", "R@")) or c in ("base_rate", "AP")
    ]
    print("\nPer query:")
    per_query = results.pivot(
        index="query", columns="model", values=[f"P@{min(args.ks)}", "AP"]
    )
    per_query.insert(0, "base_rate", results.groupby("query")["base_rate"].first())
    print(
        per_query.loc[results["query"].unique()].to_string(float_format="{:.2f}".format)
    )

    print(
        "\nMean over queries (base_rate = expected precision of a random ranking = random mAP):"
    )
    print(
        results.groupby("model")[metric_cols]
        .mean()
        .to_string(float_format="{:.3f}".format)
    )

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(out, index=False)
    print(f"\nFull results -> {out}")


if __name__ == "__main__":
    main()
