import json
import time

import numpy as np
from PIL import Image

from scene_clipper import semantic_search as ss


class FakeEmbedder:
    """Embeds by keyword so ranking is predictable without loading a model."""

    WORDS = ["rain", "gym", "kiss"]

    def _vec(self, text):
        v = np.asarray([float(w in text.lower()) for w in self.WORDS] + [0.1], dtype=np.float32)
        return v / np.linalg.norm(v)

    def embed_scenes(self, images, dialogues):
        return np.stack([self._vec(d) for d in dialogues])

    def embed_query(self, query):
        return self._vec(query)


def make_library(root, scenes_by_episode):
    for episode, scenes in scenes_by_episode.items():
        folder = root / episode
        folder.mkdir(parents=True)
        records = []
        for index, (start, end) in enumerate(scenes, 1):
            thumb = folder / f"scene-{index:04}.jpg"
            Image.new("RGB", (24, 14), (index * 20, 0, 0)).save(thumb)
            records.append({"id": f"{episode}-{index:04}", "episode": episode, "video": "", "start": start,
                            "end": end, "thumbnail": str(thumb)})
        (folder / "catalog.json").write_text(json.dumps({"episode": episode, "video": "", "scenes": records}))


def test_pick_dialogue_track_skips_signs_and_prefers_full_subs():
    tracks = [
        {"track": 0, "language": "eng", "title": "English Signs/Songs", "text_based": True},
        {"track": 1, "language": "eng", "title": "English (SDH)", "text_based": True},
        {"track": 2, "language": "eng", "title": "English [whomst]", "text_based": True},
        {"track": 3, "language": "jpn", "title": "Japanese", "text_based": True},
    ]
    assert ss.pick_dialogue_track(tracks)["track"] == 2
    assert ss.pick_dialogue_track(tracks[:2])["track"] == 1
    assert ss.pick_dialogue_track([tracks[3]]) is None


def test_dialogue_for_collects_overlapping_cues_once():
    cues = [
        {"start": 0, "end": 2, "text": "Before."},
        {"start": 3, "end": 4, "text": "{=74}Are you going to the gym?"},
        {"start": 4.5, "end": 6, "text": "Are you going to the gym?"},
        {"start": 5, "end": 7, "text": "Yes.\\NToday."},
        {"start": 9, "end": 10, "text": "After."},
    ]
    assert ss.dialogue_for(cues, 2.5, 8) == "Are you going to the gym? Yes. Today."


def test_scene_document_interleaves_dialogue_with_image():
    image = Image.new("RGB", (4, 4))
    assert ss.scene_document(image, "") is image
    doc = ss.scene_document(image, "hello")
    assert doc["image"] == [image] and "<|image|>" in doc["text"] and "hello" in doc["text"]


def test_build_index_then_search_ranks_across_episodes(tmp_path, monkeypatch):
    library = tmp_path / "lib"
    make_library(library, {"E01": [(0, 4), (4, 9)], "E02": [(0, 5)]})
    lines = {("E01", 0): "walking in the rain", ("E01", 4): "at the gym", ("E02", 0): "a kiss in the rain"}
    monkeypatch.setattr(ss, "dialogue_cues", lambda video, cache: [])
    monkeypatch.setattr(ss, "dialogue_for", lambda cues, start, end: "")
    embedder = FakeEmbedder()
    real = embedder.embed_scenes
    calls = []

    def embed_scenes(images, dialogues):
        calls.append(len(images))
        episode = "E01" if len(calls) == 1 else "E02"
        starts = [0, 4] if episode == "E01" else [0]
        return real(images, [lines[(episode, s)] for s in starts])

    embedder.embed_scenes = embed_scenes
    assert ss.index_status(library) == {"episode_count": 2, "indexed_count": 0, "ready": False}
    result = ss.build_index(library, tmp_path / "subs", embedder)
    assert result == {"indexed": 2, "skipped": 0, "scenes": 3}
    assert ss.index_status(library)["ready"]

    hits = ss.search(library, "rain", embedder, limit=2)
    assert [h["id"] for h in hits] == ["E01-0001", "E02-0001"]
    assert ss.search(library, "gym", embedder, limit=1)[0]["id"] == "E01-0002"

    # Second build is a no-op until a catalog changes.
    assert ss.build_index(library, tmp_path / "subs", embedder)["skipped"] == 2
    time.sleep(0.01)
    catalog = library / "E02" / "catalog.json"
    catalog.write_text(catalog.read_text())
    assert ss.index_status(library)["indexed_count"] == 1


def test_search_drops_near_duplicate_hits(tmp_path):
    library = tmp_path / "lib"
    make_library(library, {"E01": [(0, 4), (4, 9), (9, 14)]})
    embedder = FakeEmbedder()
    embedder.embed_scenes = lambda images, _d: np.stack([embedder._vec(t) for t in ["rain", "rain", "rain gym"]])
    ss.build_index(library, tmp_path / "subs", embedder)
    assert [h["id"] for h in ss.search(library, "rain", embedder)] == ["E01-0001", "E01-0003"]
    assert len(ss.search(library, "rain", embedder, duplicate_similarity=1.1)) == 3
