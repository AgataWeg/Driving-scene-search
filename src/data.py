"""Load nuScenes front-camera keyframes and derive labels from annotations.

Produces one row per frame:
frame_id, image_path, scene, description, time_of_day, weather, objects
"""
from pathlib import Path

import pandas as pd
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import BoxVisibility

# nuScenes category prefix -> simple label used in queries
CATEGORY_MAP = {
    "human.pedestrian": "pedestrian",
    "vehicle.car": "car",
    "vehicle.truck": "truck",
    "vehicle.bus": "bus",
    "vehicle.bicycle": "bicycle",
    "vehicle.motorcycle": "motorcycle",
    "vehicle.construction": "construction_vehicle",
    "movable_object.trafficcone": "traffic_cone",
    "movable_object.barrier": "barrier",
}

MIN_VISIBILITY = 3  # nuScenes visibility bins 1-4; 3 = 60-80% visible, 4 = 80-100%


def simple_label(category_name: str) -> str | None:
    for prefix, label in CATEGORY_MAP.items():
        if category_name.startswith(prefix):
            return label
    return None


def load_frames(dataroot: str, version: str = "v1.0-mini", camera: str = "CAM_FRONT") -> pd.DataFrame:
    nusc = NuScenes(version=version, dataroot=dataroot, verbose=False)
    rows = []
    for sample in nusc.sample:
        scene = nusc.get("scene", sample["scene_token"])
        desc = scene["description"].lower()
        cam_token = sample["data"][camera]

        # Boxes that actually fall inside this camera's image
        path, boxes, _ = nusc.get_sample_data(cam_token, box_vis_level=BoxVisibility.ANY)

        objects = set()
        for box in boxes:
            ann = nusc.get("sample_annotation", box.token)
            if int(ann["visibility_token"]) < MIN_VISIBILITY:
                continue
            label = simple_label(box.name)
            if label:
                objects.add(label)

        rows.append({
            "frame_id": cam_token,
            "image_path": str(Path(path)),
            "scene": scene["name"],
            "description": desc,
            "time_of_day": "night" if "night" in desc else "day",
            "weather": "rain" if "rain" in desc else "clear",
            "objects": ",".join(sorted(objects)),
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--dataroot", required=True)
    p.add_argument("--out", default="data/frames.csv")
    args = p.parse_args()

    df = load_frames(args.dataroot)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"{len(df)} frames -> {args.out}")
    print(df[["time_of_day", "weather"]].value_counts())
