# Driving scene search

Text-to-image search over nuScenes camera frames using vision-language foundation models (CLIP, SigLIP) and FAISS.

Work in progress. Planned: recall@k evaluation, CLIP vs. SigLIP comparison, an LLM query-rewriting step.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Download **nuScenes mini** (v1.0-mini) from https://www.nuscenes.org/nuscenes#download (free account needed) and unpack it, e.g. to `data\nuscenes` in the project directory.

## Run

```bash
# 1. Frame table with labels derived from annotations
python -m src.data --dataroot .\data\nuscenes

# 2. Embed frames and build the index (a few minutes on CPU, run once per model)
python -m scripts.build_index --model clip
python -m scripts.build_index --model siglip

# 3. Search
python -m scripts.search_cli --model clip "a pedestrian on a street at night"
```

## How it works

Both models have an image encoder and a text encoder trained so that an image and a matching caption map to nearby vectors. All frames are embedded once and stored in a FAISS index. A text query is embedded with the same model's text encoder, and the frames with the highest cosine similarity are returned.

## Labels

`src/data.py` derives labels without manual annotation:
- `time_of_day` and `weather` from the scene description ("night", "rain")
- `objects` from 3D box annotations visible in the camera (visibility ≥ 60%)

These are used later as ground truth for recall@k.