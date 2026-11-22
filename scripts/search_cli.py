"""Search indexed frames from the terminal."""

import argparse

from src.embed import get_embedder
from src.search import Searcher
from src.rewrite import rewrite_query
from dotenv import load_dotenv


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query")
    parser.add_argument("--model", choices=["clip", "siglip"], default="clip")
    parser.add_argument("--frames", default="data/frames.csv")
    parser.add_argument("-k", type=int, default=10)
    parser.add_argument(
        "--rewrite",
        action="store_true",
        help="rewrite the query with an LLM before searching",
    )
    args = parser.parse_args()
    load_dotenv()

    if args.rewrite:
        query = rewrite_query(args.query)
        print(f"Rewritten query: {query}\n")
    else:
        query = args.query

    searcher = Searcher(get_embedder(args.model), f"indexes/{args.model}", args.frames)
    results = searcher.search(query, args.k)
    print(
        results[
            ["score", "scene", "time_of_day", "weather", "objects", "image_path"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
