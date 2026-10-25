"""CLIP and SigLIP wrappers with one shared interface.

Both return L2-normalized float32 vectors, so a dot product = cosine similarity.
Runs on CPU; use small batches and torch.no_grad().
"""
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

torch.set_num_threads(max(1, torch.get_num_threads()))


def _normalize(x: torch.Tensor) -> np.ndarray:
    x = x / x.norm(dim=-1, keepdim=True)
    return x.cpu().numpy().astype("float32")


def _pooled_features(output: object) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, tuple):
        if len(output) < 2 or not isinstance(output[1], torch.Tensor):
            raise TypeError("Expected pooled features in the model output tuple")
        return output[1]
    pooled = getattr(output, "pooler_output", None)
    if not isinstance(pooled, torch.Tensor):
        raise TypeError("Expected model output to contain pooled features")
    return pooled


class Embedder:
    name: str

    def encode_images(self, paths: list[str], batch_size: int = 16) -> np.ndarray:
        raise NotImplementedError

    def encode_text(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError


class ClipEmbedder(Embedder):
    def __init__(self, model_name: str = "ViT-B-32", pretrained: str = "laion2b_s34b_b79k"):
        import open_clip

        self.name = f"clip-{model_name}"
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model.eval()

    @torch.no_grad()
    def encode_images(self, paths, batch_size=16):
        out = []
        for i in tqdm(range(0, len(paths), batch_size), desc=f"{self.name} images"):
            batch = torch.stack([self.preprocess(Image.open(p).convert("RGB")) for p in paths[i:i + batch_size]])
            out.append(_normalize(self.model.encode_image(batch)))
        return np.concatenate(out)

    @torch.no_grad()
    def encode_text(self, texts):
        return _normalize(self.model.encode_text(self.tokenizer(texts)))


class SiglipEmbedder(Embedder):
    def __init__(self, model_id: str = "google/siglip-base-patch16-224"):
        from transformers import AutoModel, AutoProcessor

        self.name = "siglip-base"
        self.model = AutoModel.from_pretrained(model_id).eval()
        self.processor = AutoProcessor.from_pretrained(model_id)

    @torch.no_grad()
    def encode_images(self, paths, batch_size=16):
        out = []
        for i in tqdm(range(0, len(paths), batch_size), desc=f"{self.name} images"):
            imgs = [Image.open(p).convert("RGB") for p in paths[i:i + batch_size]]
            inputs = self.processor(images=imgs, return_tensors="pt")
            out.append(_normalize(_pooled_features(self.model.get_image_features(**inputs))))
        return np.concatenate(out)

    @torch.no_grad()
    def encode_text(self, texts):
        # SigLIP was trained with max_length padding; other padding hurts results
        inputs = self.processor(text=texts, padding="max_length", return_tensors="pt")
        return _normalize(_pooled_features(self.model.get_text_features(**inputs)))


def get_embedder(name: str) -> Embedder:
    if name == "clip":
        return ClipEmbedder()
    if name == "siglip":
        return SiglipEmbedder()
    raise ValueError(f"Unknown model: {name} (use 'clip' or 'siglip')")
