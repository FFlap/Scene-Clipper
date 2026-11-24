"""Text-to-scene search backed by EmbeddingGemma 2.

Each scene is embedded once (thumbnail + the dialogue spoken during it) into a
per-episode ``search.npz``. A query is embedded with the SearchQuery prompt and
ranked against every scene in the library with a single dot product.
"""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image

from .subtitles import _cache_key, parse_srt, probe_subtitle_tracks

MODEL_NAME = "google/embeddinggemma-2"
INDEX_FILE = "search.npz"
INDEX_VERSION = 1
# Thumbnails are ~240px wide, so the default 280-token image budget only upscales.
# 140 tokens is ~2.3x faster with embeddings ~0.99 cosine to the default.
VISION_TOKENS = 140
MAX_DIALOGUE_CHARS = 600


class GemmaSearchEmbedder:
    """Lazily loads EmbeddingGemma 2 (text + vision encoders only)."""

    def __init__(self, device: str | None = None, vision_tokens: int = VISION_TOKENS):
        self.device = device
        self.vision_tokens = vision_tokens
        self.model = None

    def _load(self):
        if self.model is not None:
            return self.model
        import torch
        from sentence_transformers import SentenceTransformer

        device = self.device or ("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
        # float16 yields NaN/degraded embeddings with this model; bf16 or fp32 only.
        dtype = torch.float32 if device == "cpu" else torch.bfloat16
        model = SentenceTransformer(
            MODEL_NAME, device=device,
            model_kwargs={"torch_dtype": dtype},
            config_kwargs={"audio_config": None},
        )
        processor = getattr(model[0], "processor", None)
        if processor is not None and getattr(processor, "image_processor", None) is not None:
            processor.image_processor.max_soft_tokens = self.vision_tokens
            processor.image_processor.image_seq_length = self.vision_tokens
            processor.image_seq_length = self.vision_tokens
        self.model = model
        return model

    def embed_scenes(self, images, dialogues, batch_size: int = 8):
        model = self._load()
        docs = [scene_document(image, text) for image, text in zip(images, dialogues)]
        values = model.encode(docs, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(values, dtype=np.float32)

    def embed_query(self, query: str):
        model = self._load()
        value = model.encode([query], prompt_name="SearchQuery", normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(value, dtype=np.float32)[0]


def scene_document(image, dialogue: str = ""):
    dialogue = (dialogue or "").strip()
    if not dialogue:
        return image
    return {"text": f"title: none | text: <|image|>\nDialogue: {dialogue}", "image": [image]}


# ---------- dialogue ----------

def pick_dialogue_track(tracks, language: str = "eng"):
    """Choose the full-dialogue text track, skipping signs/songs-only tracks."""
    candidates = [t for t in tracks if t.get("text_based") and t.get("language") == language]
    if not candidates:
        return None

    def rank(track):
        title = track.get("title", "").lower()
        if "sign" in title or "song" in title or "forced" in title:
            return 3
        if "sdh" in title or "cc" in title:
            return 1
        return 0

    return min(candidates, key=lambda t: (rank(t), t["track"]))


def dialogue_cues(video: Path, cache_dir: Path, language: str = "eng"):
    """Cues for the main dialogue track, reusing the subtitle-search cache when possible."""
    video = Path(video)
    cache_dir = Path(cache_dir)
    shared = cache_dir / _cache_key(video) / "meta.json"
    if shared.exists():
        try:
            meta = json.loads(shared.read_text(encoding="utf-8"))
            track = pick_dialogue_track(meta.get("tracks", []), language)
            if track and (shared.parent / track["srt"]).exists():
                return parse_srt((shared.parent / track["srt"]).read_text(encoding="utf-8", errors="replace"))
        except (json.JSONDecodeError, OSError):
            pass
    own = cache_dir / _cache_key(video) / f"search-dialogue-{language}.srt"
    if not own.exists():
        try:
            track = pick_dialogue_track(probe_subtitle_tracks(video), language)
        except (subprocess.CalledProcessError, OSError):
            return []
        if track is None:
            return []
        own.parent.mkdir(parents=True, exist_ok=True)
        partial = own.with_suffix(".partial.srt")
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(video),
                 "-map", f"0:s:{track['track']}", "-c:s", "srt", str(partial)],
                check=True, capture_output=True, text=True,
            )
        except (subprocess.CalledProcessError, OSError):
            return []
        partial.rename(own)
    return parse_srt(own.read_text(encoding="utf-8", errors="replace"))


def dialogue_for(cues, start: float, end: float) -> str:
    seen, parts = set(), []
    for cue in cues:
        if cue["end"] >= start and cue["start"] <= end:
            text = re.sub(r"\{[^}]*\}", "", re.sub(r"\\[Nn]", " ", cue["text"])).strip()
            if text and text not in seen:
                seen.add(text)
                parts.append(text)
    return " ".join(parts)[:MAX_DIALOGUE_CHARS]


# ---------- index ----------

def episode_catalogs(library: Path):
    return sorted(Path(library).glob("*/catalog.json"))


def index_status(library: Path):
    catalogs = episode_catalogs(library)
    indexed = sum(1 for c in catalogs if _index_current(c))
    return {"episode_count": len(catalogs), "indexed_count": indexed, "ready": bool(catalogs) and indexed == len(catalogs)}


def _index_current(catalog_path: Path) -> bool:
    index = catalog_path.with_name(INDEX_FILE)
    if not index.exists() or index.stat().st_mtime < catalog_path.stat().st_mtime:
        return False
    try:
        with np.load(index) as data:
            return int(data["version"]) == INDEX_VERSION
    except (OSError, KeyError, ValueError):
        return False


def build_index(library: Path, subtitle_cache: Path, embedder=None, rebuild: bool = False, progress=None):
    """Embed every scene in a library. Returns {"indexed": n, "skipped": n, "scenes": n}."""
    embedder = embedder or GemmaSearchEmbedder()
    catalogs = episode_catalogs(library)
    indexed = skipped = scene_total = 0
    for number, catalog_path in enumerate(catalogs, 1):
        if not rebuild and _index_current(catalog_path):
            skipped += 1
            continue
        payload = json.loads(catalog_path.read_text(encoding="utf-8"))
        scenes = payload.get("scenes", [])
        if progress:
            progress(number, len(catalogs), payload.get("episode", catalog_path.parent.name))
        cues = dialogue_cues(Path(payload["video"]), subtitle_cache) if payload.get("video") and Path(payload["video"]).exists() else []
        images, dialogues = [], []
        for scene in scenes:
            with Image.open(scene["thumbnail"]) as loaded:
                images.append(loaded.convert("RGB"))
            dialogues.append(dialogue_for(cues, scene["start"], scene["end"]))
        # Embed in chunks so a live search can grab the model between them.
        chunks = [embedder.embed_scenes(images[i:i + 32], dialogues[i:i + 32]) for i in range(0, len(images), 32)]
        embeddings = np.concatenate(chunks) if chunks else np.zeros((0, 768), dtype=np.float32)
        partial = catalog_path.with_name("search.partial.npz")
        np.savez_compressed(
            partial,
            version=np.asarray(INDEX_VERSION),
            ids=np.asarray([s["id"] for s in scenes]),
            embeddings=np.asarray(embeddings, dtype=np.float16),
            dialogue=np.asarray(dialogues),
        )
        partial.replace(catalog_path.with_name(INDEX_FILE))
        indexed += 1
        scene_total += len(scenes)
    return {"indexed": indexed, "skipped": skipped, "scenes": scene_total}


def search(library: Path, query: str, embedder, limit: int = 60, duplicate_similarity: float = 0.97):
    """Rank scenes for a query, dropping hits nearly identical to a better one (e.g. OP/ED shots repeated every episode)."""
    ids, vectors, dialogues = [], [], []
    for catalog_path in episode_catalogs(library):
        index = catalog_path.with_name(INDEX_FILE)
        if not index.exists():
            continue
        with np.load(index) as data:
            ids.extend(data["ids"].tolist())
            vectors.append(data["embeddings"].astype(np.float32))
            dialogues.extend(data["dialogue"].tolist())
    if not ids:
        return []
    matrix = np.concatenate(vectors)
    scores = matrix @ embedder.embed_query(query)
    kept = []
    for i in np.argsort(-scores):
        if kept and float(np.max(matrix[kept] @ matrix[i])) >= duplicate_similarity:
            continue
        kept.append(int(i))
        if len(kept) >= limit:
            break
    return [{"id": ids[i], "score": float(scores[i]), "dialogue": dialogues[i]} for i in kept]
