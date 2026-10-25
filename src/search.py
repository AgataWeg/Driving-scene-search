"""Text -> top-k frames."""
from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from src.index import load

if TYPE_CHECKING:
    from src.embed import Embedder


class Searcher:
    def __init__(self, embedder: Embedder, index_dir: str, frames_csv: str):
        self.embedder = embedder
        self.index, self.frame_ids = load(index_dir)
        self.frames = pd.read_csv(frames_csv).set_index("frame_id")

    def search(self, text: str, k: int = 10) -> pd.DataFrame:
        q = self.embedder.encode_text([text])
        scores, idx = self.index.search(q, k)
        ids = [self.frame_ids[i] for i in idx[0]]
        result = self.frames.loc[ids].reset_index()
        result.insert(1, "score", scores[0])
        return result
