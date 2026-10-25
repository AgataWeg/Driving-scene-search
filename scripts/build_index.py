"""Embed dataset frames and build a FAISS index."""
import argparse

import pandas as pd

from src.embed import get_embedder
from src.index import build_index, save


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["clip", "siglip"], default="clip")
    parser.add_argument("--frames", default="data/frames.csv")
    parser.add_argument("--out")
    args = parser.parse_args()

    frames = pd.read_csv(args.frames)
    required_columns = {"frame_id", "image_path"}
    missing_columns = required_columns.difference(frames.columns)
    if missing_columns:
        raise ValueError(f"{args.frames} is missing required columns: {', '.join(sorted(missing_columns))}")

    frame_ids = frames["frame_id"].astype(str).tolist()
    image_paths = frames["image_path"].tolist()
    embedder = get_embedder(args.model)
    vectors = embedder.encode_images(image_paths)
    index = build_index(vectors)

    out_dir = args.out or f"indexes/{args.model}"
    save(index, vectors, frame_ids, out_dir)
    print(f"{len(frame_ids)} frames indexed -> {out_dir}")


if __name__ == "__main__":
    main()