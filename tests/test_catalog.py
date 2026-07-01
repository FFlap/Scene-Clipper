from scene_clipper import catalog
from scene_clipper.catalog import SceneItem, filter_and_group, find_videos, similarity_threshold


def test_filters_short_and_groups_similar_scenes():
    items = [
        SceneItem("e1", 0, 2, 1, 1, (10, 10, 10), "a.jpg"),
        SceneItem("e1", 2, 6, 4, 1, (10, 10, 10), "b.jpg"),
        SceneItem("e1", 6, 10, 8, 3, (12, 12, 12), "c.jpg"),
        SceneItem("e1", 10, 15, 12.5, 2**63, (240, 240, 240), "d.jpg"),
    ]
    result = filter_and_group(items, min_duration=3, hash_distance=18, color_distance=70)
    assert len(result) == 2
    assert result[0].similar_scene_count == 2
    assert result[0].thumbnail == "b.jpg"


def test_groups_globally_by_semantic_embedding():
    items = [
        SceneItem("e",0,4,2,1,(0,0,0),"a",embedding=(1.0,0.0)),
        SceneItem("e",100,104,102,2**63,(255,255,255),"b",embedding=(0.8,0.6)),
    ]
    assert len(filter_and_group(items, semantic_similarity=0.74)) == 1


def test_similarity_slider_maps_conservative_to_aggressive():
    assert similarity_threshold(0) > 1
    assert similarity_threshold(100) == 0.65


def test_find_videos_ignores_macos_appledouble_sidecars(tmp_path):
    real = tmp_path / "episode.mkv"
    sidecar = tmp_path / "._episode.mkv"
    real.write_bytes(b"real")
    sidecar.write_bytes(b"not a video")

    assert find_videos(tmp_path) == [real]


def test_preprocess_directory_can_continue_after_bad_video(monkeypatch, tmp_path):
    good = tmp_path / "good.mkv"
    bad = tmp_path / "bad.mkv"
    good.write_bytes(b"good")
    bad.write_bytes(b"bad")

    monkeypatch.setattr(catalog, "find_videos", lambda _: [good, bad])

    def fake_preprocess(video, output, rebuild, embedder, *, id_prefix=""):
        if video == bad:
            raise RuntimeError("broken")
        return {"episode": "good", "representative_count": 1, "id_prefix": id_prefix}

    monkeypatch.setattr(catalog, "preprocess_episode", fake_preprocess)

    result = catalog.preprocess_directory(tmp_path, tmp_path / "out", continue_on_error=True, id_prefix="lib")

    assert result["results"] == [{"episode": "good", "representative_count": 1, "id_prefix": "lib"}]
    assert result["failures"] == [{"video": str(bad), "error": "broken"}]
