import json
import time
from scene_clipper.review_app import create_app


def test_catalog_and_export_routes(tmp_path):
    catalog_dir=tmp_path/"catalog"/"E01"; catalog_dir.mkdir(parents=True)
    scene={"id":"x","episode":"E01","video":"a.mkv","start":1,"end":5,"thumbnail":"x.jpg"}
    (catalog_dir/"catalog.json").write_text(json.dumps({"episode":"E01","scenes":[scene]}))
    calls=[]
    def exporter(selected,root,clips): calls.append((selected,clips)); out=root/"done"; out.mkdir(parents=True); return out
    client=create_app(tmp_path/"catalog",tmp_path/"exports",exporter).test_client()
    assert client.get("/api/catalog").json[0]["episode"]=="E01"
    assert client.post("/api/export",json={"selected":[],"generate_mp4":False}).status_code==400
    response=client.post("/api/export",json={"selected":["x"],"generate_mp4":True})
    assert response.status_code==200 and calls[0][1] is True


def test_export_uses_unique_scene_ids_across_libraries(tmp_path):
    root = tmp_path / "catalog"
    for library, video in [("A", "a.mkv"), ("B", "b.mkv")]:
        catalog_dir = root / library / "E01"
        catalog_dir.mkdir(parents=True)
        scene = {"id": f"{library}:E01-0001", "episode": "E01", "video": video, "start": 1, "end": 5, "thumbnail": "x.jpg"}
        (catalog_dir / "catalog.json").write_text(json.dumps({"episode": "E01", "scenes": [scene]}))
    calls = []
    def exporter(selected, root, clips):
        calls.append(selected)
        out = root / "done"
        out.mkdir(parents=True)
        return out

    client = create_app(root, tmp_path / "exports", exporter).test_client()
    response = client.post("/api/export", json={"selected": ["B:E01-0001"], "generate_mp4": False})

    assert response.status_code == 200
    assert calls[0][0]["video"] == "b.mkv"


def test_preprocess_reports_partial_failures(monkeypatch, tmp_path):
    video_dir = tmp_path / "videos"
    video_dir.mkdir()
    good = video_dir / "good.mkv"
    bad = video_dir / "bad.mkv"
    good.write_bytes(b"good")
    bad.write_bytes(b"bad")

    def fake_find_videos(source):
        return [good, bad]

    def fake_preprocess_directory(source, target, rebuild, *, continue_on_error, id_prefix):
        return {"results": [{"representative_count": 3}], "failures": [{"video": str(bad), "error": "broken"}]}

    monkeypatch.setattr("scene_clipper.review_app.find_videos", fake_find_videos)
    monkeypatch.setattr("scene_clipper.review_app.preprocess_directory", fake_preprocess_directory)

    client = create_app(tmp_path / "catalog", tmp_path / "exports").test_client()
    response = client.post("/api/preprocess", json={"path": str(video_dir), "name": "Videos"})

    assert response.status_code == 200
    for _ in range(20):
        status = client.get("/api/preprocess/status").json[0]
        if status["status"] == "done_with_errors":
            break
        time.sleep(0.01)
    assert status["status"] == "done_with_errors"
    assert status["failures"][0]["error"] == "broken"
