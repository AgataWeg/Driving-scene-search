"""Build, save and load a FAISS index plus the frame IDs it refers to."""
from pathlib import Path

import faiss
import numpy as np


def build_index(vectors: np.ndarray) -> faiss.Index:
    # Inner product on normalized vectors = cosine similarity.
    # Exact (flat) search is fine for a few thousand frames.
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index


def save(index: faiss.Index, vectors: np.ndarray, frame_ids: list[str], out_dir: str) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(out / "index.faiss"))
    np.save(out / "vectors.npy", vectors)
    (out / "frame_ids.txt").write_text("\n".join(frame_ids))


def load(out_dir: str) -> tuple[faiss.Index, list[str]]:
    out = Path(out_dir)
    index = faiss.read_index(str(out / "index.faiss"))
    frame_ids = (out / "frame_ids.txt").read_text().splitlines()
    return index, frame_ids
