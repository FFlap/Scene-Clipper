import json
import time
import subprocess
from scene_clipper.review_app import create_app, ffmpeg_preview_command


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


def test_media_tracks_reports_audio_languages(monkeypatch, tmp_path):
    source = tmp_path / "episodes"
    source.mkdir()
    video = source / "E01.mkv"
    video.write_bytes(b"video")
    monkeypatch.setattr(
        "scene_clipper.review_app.subprocess.run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 0, '{"streams":[{"index":1,"tags":{"language":"jpn","title":"Japanese"}},{"index":2,"tags":{"language":"eng"}}]}', ""),
    )
    client = create_app(tmp_path / "catalog", tmp_path / "exports").test_client()

    response = client.get("/api/media/tracks", query_string={"video": str(video)})

    assert response.status_code == 200
    assert response.json == [
        {"track": 0, "stream_index": 1, "language": "jpn", "title": "Japanese"},
        {"track": 1, "stream_index": 2, "language": "eng", "title": ""},
    ]


def test_clip_passes_selected_tracks_to_ffmpeg(monkeypatch, tmp_path):
    source = tmp_path / "episodes"
    source.mkdir()
    video = source / "E01.mkv"
    video.write_bytes(b"video")
    calls = []
    monkeypatch.setattr("scene_clipper.review_app.subprocess.run", lambda cmd, **kwargs: calls.append(cmd))
    client = create_app(tmp_path / "catalog", tmp_path / "exports").test_client()

    response = client.post("/api/clip", json={
        "video": str(video), "start": 1, "end": 3,
        "audio_track": 1, "subtitle_track": 2, "include_subtitles": True,
    })

    assert response.status_code == 200
    assert "0:a:1" in calls[0]
    assert "0:s:2" in calls[0]


def test_clip_audio_control_starts_hidden_outside_a_folder(tmp_path):
    client = create_app(tmp_path / "catalog", tmp_path / "exports").test_client()

    page = client.get("/").get_data(as_text=True)

    assert 'id="openSearchModal" hidden' in page


def test_preview_command_maps_selected_audio_track():
    cmd = ffmpeg_preview_command("a.mkv", 2, 8, "preview.mp4", audio_track=1)

    assert "0:a:1" in cmd


def test_audio_selector_reloads_preview(tmp_path):
    client = create_app(tmp_path / "catalog", tmp_path / "exports").test_client()

    page = client.get("/").get_data(as_text=True)

    assert "params.set('audio_track',audioTrack)" in page
    assert "#audioTrack').onchange" in page


def test_clip_editor_has_reset_and_loop_controls(tmp_path):
    page = create_app(tmp_path / "catalog", tmp_path / "exports").test_client().get("/").get_data(as_text=True)

    assert 'id="resetRange"' in page
    assert 'id="loopRange"' in page
    assert 'id="setIn"' not in page
    assert 'id="setOut"' not in page
    assert 'id="playRange"' not in page


def test_subtitle_overlay_follows_include_toggle(tmp_path):
    page = create_app(tmp_path / "catalog", tmp_path / "exports").test_client().get("/").get_data(as_text=True)

    assert "includeSubtitles').checked?cue.text:''" in page
    assert "includeSubtitles').onchange" in page
    assert "querySelector('#cueList').hidden" not in page
    assert "querySelector('#tlCues').hidden" not in page
