"""Search indexed frames from the terminal."""
import argparse

from src.embed import get_embedder
from src.search import Searcher


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query")
    parser.add_argument("--model", choices=["clip", "siglip"], default="clip")
    parser.add_argument("--frames", default="data/frames.csv")
    parser.add_argument("-k", type=int, default=10)
    args = parser.parse_args()

    searcher = Searcher(get_embedder(args.model), f"indexes/{args.model}", args.frames)
    results = searcher.search(args.query, args.k)
    print(results[["score", "scene", "time_of_day", "weather", "objects", "image_path"]].to_string(index=False))


if __name__ == "__main__":
    main()
