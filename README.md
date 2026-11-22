# Driving scene search

Text-to-image search over nuScenes camera frames using vision-language foundation models (CLIP, SigLIP) and FAISS.

Includes a recall@k evaluation, a CLIP vs. SigLIP comparison, and an optional LLM query-rewriting step.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Download **nuScenes mini** (v1.0-mini) from https://www.nuscenes.org/nuscenes#download (free account needed) and unpack it, e.g. to `data\nuscenes` in the project directory.

Optional, only for `--rewrite`: create a `.env` file in the project directory with your Anthropic API key:

```
ANTHROPIC_API_KEY=sk-ant-...
```

## Run

```bash
# 1. Frame table with labels derived from annotations
python -m src.data --dataroot .\data\nuscenes

# 2. Embed frames and build the index (a few minutes on CPU, run once per model)
python -m scripts.build_index --model clip
python -m scripts.build_index --model siglip

# 3. Search
python -m scripts.search_cli --model clip "a pedestrian on a street at night"

# 4. Evaluate and compare models (uses the built indexes; writes results/eval.csv)
python -m scripts.evaluate

# Optional: rewrite queries with an LLM first (needs ANTHROPIC_API_KEY)
python -m scripts.search_cli --rewrite "rainy bus at night"
python -m scripts.evaluate --rewrite   # writes results/eval_rewrite.csv
```

## How it works

Both models have an image encoder and a text encoder trained so that an image and a matching caption map to nearby vectors. All frames are embedded once and stored in a FAISS index. A text query is embedded with the same model's text encoder, and the frames with the highest cosine similarity are returned.

## Labels

`src/data.py` derives labels without manual annotation:
- `time_of_day` and `weather` from the scene description ("night", "rain")
- `objects` from 3D box annotations visible in the camera (visibility ≥ 60%)

These are used as ground truth in the evaluation.

## Evaluation

`src/evaluate.py` defines benchmark queries, each with label conditions (e.g. night + pedestrian). A frame is relevant if it meets all conditions. Each query ranks all frames and reports:
- `P@k`: share of the top k that are relevant
- `R@k`: share of all relevant frames found in the top k (low for common labels by construction)
- `AP`: average precision over the full ranking; the mean over queries is mAP
- `base_rate`: share of relevant frames in the dataset, i.e. what a random ranking would score. Compare against this, since some labels (e.g. `car`) appear in almost every frame.

## LLM query rewriting (optional)

CLIP and SigLIP were trained on short image captions, so a plain user query like "rainy bus at night" may not match them well. With `--rewrite`, `src/rewrite.py` first asks an LLM (Claude Haiku) to turn the query into a visual caption and searches with that. Examples from an actual run:

| Query | Rewritten |
|---|---|
| a bus at night in the rain | a photo of a bus driving on a wet city road at night in the rain, with headlights and reflections on the wet asphalt |
| a cyclist during the day | a photo of a cyclist riding on a city street during the day, seen from a car's front camera |

The evaluation still checks results against the labels of the original query, so comparing `eval.csv` with `eval_rewrite.csv` shows whether rewriting helps. The rewritten text is saved in the `search_text` column.

Without the flag, no API key or LLM call is needed.

## Results

15 benchmark queries on nuScenes mini (404 front-camera frames), mean over queries:

| Setup | P@10 | mAP |
|---|---|---|
| Random ranking | 0.22 | 0.22 |
| CLIP | 0.53 | 0.40 |
| SigLIP | 0.61 | 0.44 |
| CLIP + rewrite | 0.61 | 0.44 |
| SigLIP + rewrite | 0.61 | 0.44 |

- **Both models are well above random.** Day vs. night, construction sites and traffic cones are easy (AP up to 0.99). Rain, bicycles and rare combinations such as "a bus at night in the rain" (2% of frames) remain hard for every setup.
- **SigLIP beats CLIP on plain queries** (mAP 0.44 vs. 0.40), mainly on trucks, rain and pedestrians at night.
- **Rewriting helps CLIP but not SigLIP.** It brings CLIP up to SigLIP's level, with the largest gain on "a cyclist during the day" (AP 0.08 → 0.31). SigLIP is unchanged on average; some queries improve (bus 0.47 → 0.58) and others get worse (truck 0.38 → 0.30, road barriers 0.24 → 0.15). A likely reason is that CLIP is known to be sensitive to caption-style prompts like "a photo of ...", which the rewrite adds.

**Caveats:** the dataset has only 10 scenes, and frames from the same scene look very similar, so differences of a few points are not reliable. Labels are derived automatically from scene descriptions and 3D boxes, so some "misses" may be correct images with missing labels. LLM rewrites can vary between runs.
