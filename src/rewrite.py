"""Optional LLM query rewriting: turn a user query into a CLIP/SigLIP-style caption.

Requires the `anthropic` package and the ANTHROPIC_API_KEY environment variable.
The import is inside the function, so the rest of the project works without them.
"""

from functools import lru_cache

MODEL = "claude-haiku-5-5"

SYSTEM_PROMPT = """You rewrite search queries for a text-to-image search engine over \
front-camera images from a car driving in a city.

The engine uses CLIP/SigLIP, which match images to short descriptive captions. \
Rewrite the user's query into one such caption:
- Start with "a photo of".
- Describe only what is visible: objects, road, time of day, lighting, weather.
- Keep everything the user asked for; you may add visual cues that follow from it \
(e.g. night -> dark sky, headlights; rain -> wet road, reflections).
- Do not use negations such as "no cars".
- Keep it under 25 words.

Reply with the caption only, on a single line, with no quotes or explanation."""


@lru_cache(maxsize=256)
def rewrite_query(text: str) -> str:
    import anthropic

    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
    response = client.messages.create(
        model=MODEL,
        max_tokens=100,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": text}],
    )
    return response.content[0].text.strip()
