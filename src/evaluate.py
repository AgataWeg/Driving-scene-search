"""Retrieval evaluation against labels derived in src/data.py.

Each benchmark query has a text prompt and label conditions. A frame is relevant
if it satisfies every condition, e.g. {"time_of_day": "night", "objects": ["pedestrian"]}.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.search import Searcher


@dataclass
class Query:
    text: str
    time_of_day: str | None = None
    weather: str | None = None
    objects: list[str] = field(default_factory=list)


# Mix of single-attribute and combined queries. Note: in nuScenes mini all rain frames are also night.
QUERIES = [
    Query("a city street at night", time_of_day="night"),
    Query("a sunny street during the day", time_of_day="day"),
    Query("driving in the rain, wet road", weather="rain"),
    Query("a pedestrian walking on the street", objects=["pedestrian"]),
    Query("a bus on the road", objects=["bus"]),
    Query("a truck on the road", objects=["truck"]),
    Query("a cyclist riding a bicycle", objects=["bicycle"]),
    Query("a motorcycle on the road", objects=["motorcycle"]),
    Query("traffic cones on the road", objects=["traffic_cone"]),
    Query("road barriers", objects=["barrier"]),
    Query("a construction site with construction vehicles", objects=["construction_vehicle"]),
    Query("a pedestrian on a street at night", time_of_day="night", objects=["pedestrian"]),
    Query("a bus at night in the rain", time_of_day="night", weather="rain", objects=["bus"]),
    Query("a cyclist during the day", time_of_day="day", objects=["bicycle"]),
    Query("traffic cones and a construction vehicle", objects=["traffic_cone", "construction_vehicle"]),
]


def relevance_mask(frames: pd.DataFrame, q: Query) -> pd.Series:
    mask = pd.Series(True, index=frames.index)
    if q.time_of_day:
        mask &= frames["time_of_day"] == q.time_of_day
    if q.weather:
        mask &= frames["weather"] == q.weather
    objects = frames["objects"].fillna("").str.split(",")
    for obj in q.objects:
        mask &= objects.apply(lambda objs, o=obj: o in objs)
    return mask


def average_precision(rel_sorted: np.ndarray) -> float:
    """AP over a full ranking: mean of precision@i at each relevant position i."""
    n_rel = rel_sorted.sum()
    if n_rel == 0:
        return float("nan")
    hits = np.cumsum(rel_sorted)
    return float((hits / np.arange(1, len(rel_sorted) + 1))[rel_sorted].sum() / n_rel)


def evaluate(searcher: Searcher, queries: list[Query] = QUERIES, ks: tuple[int, ...] = (5, 10, 50)) -> pd.DataFrame:
    """One row per query: n_relevant, base_rate, P@k, R@k for each k, and AP."""
    n_frames = len(searcher.frame_ids)
    rows = []
    for q in queries:
        # Rank every frame so AP and large-k recall are exact
        ranked = searcher.search(q.text, k=n_frames).set_index("frame_id")
        rel = relevance_mask(ranked, q).to_numpy()
        n_rel = int(rel.sum())
        row = {"query": q.text, "n_relevant": n_rel, "base_rate": n_rel / n_frames}
        for k in ks:
            hits = rel[:k].sum()
            row[f"P@{k}"] = hits / k
            row[f"R@{k}"] = hits / n_rel if n_rel else float("nan")
        row["AP"] = average_precision(rel)
        rows.append(row)
    return pd.DataFrame(rows)
