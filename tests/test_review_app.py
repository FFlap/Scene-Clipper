import json
import time
import subprocess
from scene_clipper.describe import MODEL
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


def test_export_passes_audio_language_when_selected(tmp_path):
    catalog_dir=tmp_path/"catalog"/"E01"; catalog_dir.mkdir(parents=True)
    scene={"id":"x","episode":"E01","video":"a.mkv","start":1,"end":5,"thumbnail":"x.jpg"}
    (catalog_dir/"catalog.json").write_text(json.dumps({"episode":"E01","scenes":[scene]}))
    calls=[]
    def exporter(selected,root,clips,audio_language): calls.append((clips,audio_language)); out=root/"done"; out.mkdir(parents=True); return out
    client=create_app(tmp_path/"catalog",tmp_path/"exports",exporter).test_client()

    response=client.post("/api/export",json={"selected":["x"],"generate_mp4":True,"audio_language":"jpn"})

    assert response.status_code==200
    assert calls == [(True, "jpn")]


def test_export_options_reports_selected_audio_languages(monkeypatch, tmp_path):
    catalog_dir=tmp_path/"catalog"/"E01"; catalog_dir.mkdir(parents=True)
    scene={"id":"x","episode":"E01","video":str(tmp_path/"a.mkv"),"start":1,"end":5,"thumbnail":"x.jpg"}
    (catalog_dir/"catalog.json").write_text(json.dumps({"episode":"E01","scenes":[scene]}))
    monkeypatch.setattr(
        "scene_clipper.review_app.subprocess.run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 0, '{"streams":[{"index":1,"tags":{"language":"jpn","title":"Japanese"}},{"index":2,"tags":{"language":"eng"}}]}', ""),
    )
    client=create_app(tmp_path/"catalog",tmp_path/"exports").test_client()

    response=client.post("/api/export/options",json={"selected":["x"]})

    assert response.status_code == 200
    assert response.json["languages"] == [
        {"count": 1, "language": "eng", "titles": []},
        {"count": 1, "language": "jpn", "titles": ["Japanese"]},
    ]


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


def _catalog_with_one_scene(root):
    catalog_dir = root / "E01"
    catalog_dir.mkdir(parents=True)
    scene = {"id": "x", "episode": "E01", "video": "a.mkv", "start": 1, "end": 5, "thumbnail": "x.jpg"}
    (catalog_dir / "catalog.json").write_text(json.dumps({"episode": "E01", "scenes": [scene]}))


def test_settings_round_trip_without_exposing_the_key(tmp_path):
    client = create_app(tmp_path / "catalog", tmp_path / "exports").test_client()

    assert client.get("/api/settings").json["has_api_key"] is False

    saved = client.post("/api/settings", json={"google_api_key": "AIzaSyExampleKey123"}).json

    assert saved["has_api_key"] is True
    assert "google_api_key" not in saved
    assert client.get("/api/settings").json["model"] == MODEL


def test_blank_key_field_keeps_the_stored_key_but_clearing_removes_it(tmp_path):
    client = create_app(tmp_path / "catalog", tmp_path / "exports").test_client()
    client.post("/api/settings", json={"google_api_key": "AIzaSyExampleKey123"})

    client.post("/api/settings", json={"google_api_key": ""})
    assert client.get("/api/settings").json["has_api_key"] is True

    client.post("/api/settings", json={"google_api_key": "", "clear_api_key": True})
    assert client.get("/api/settings").json["has_api_key"] is False


def test_export_builds_a_describer_from_the_saved_settings(tmp_path):
    _catalog_with_one_scene(tmp_path / "catalog")
    built = []
    exports = []

    def exporter(selected, root, clips, describer=None, **kwargs):
        exports.append(describer)
        out = root / "done"
        out.mkdir(parents=True)
        return out

    app = create_app(
        tmp_path / "catalog", tmp_path / "exports", exporter,
        describer_factory=lambda api_key: built.append(api_key) or (lambda scene: "described"),
    )
    client = app.test_client()
    client.post("/api/settings", json={"google_api_key": "AIzaSyExampleKey123"})

    response = client.post("/api/export", json={"selected": ["x"], "generate_mp4": False, "describe_clips": True})

    assert response.status_code == 200
    assert response.json["described"] is True
    assert built == ["AIzaSyExampleKey123"]
    assert exports[0] is not None


def test_export_rejects_descriptions_without_an_api_key(tmp_path):
    _catalog_with_one_scene(tmp_path / "catalog")
    calls = []

    def exporter(selected, root, clips, **kwargs):
        calls.append(kwargs)
        out = root / "done"
        out.mkdir(parents=True)
        return out

    client = create_app(tmp_path / "catalog", tmp_path / "exports", exporter).test_client()

    response = client.post("/api/export", json={"selected": ["x"], "generate_mp4": False, "describe_clips": True})

    assert response.status_code == 400
    assert "API key" in response.json["error"]
    assert calls == []


def test_settings_control_lives_only_on_the_library_screen(tmp_path):
    page = create_app(tmp_path / "catalog", tmp_path / "exports").test_client().get("/").get_data(as_text=True)
    sidebar = page.split('<aside id="sidebar">')[1].split("</aside>")[0]
    topbar = page.split('<header id="topbar">')[1].split("</header>")[0]

    assert 'id="settingsFromLibrary"' in page.split('<section id="libraryScreen">')[1]
    assert "settings" not in sidebar.lower()
    assert "settings" not in topbar.lower()
    assert 'id="describeClips"' in page
    assert "testSettingsBtn" not in page
    assert "describeByDefault" not in page


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


def test_index_page_script_parses(tmp_path):
    """A JS syntax error (e.g. a redeclared const) blanks the whole UI, so parse the inline script."""
    import re, shutil, pytest
    node = shutil.which("node")
    if not node:
        pytest.skip("node not installed")
    html = create_app(tmp_path / "catalog", tmp_path / "exports").test_client().get("/").get_data(as_text=True)
    script = tmp_path / "page.js"
    script.write_text("\n".join(re.findall(r"<script>(.*?)</script>", html, re.S)))
    result = subprocess.run([node, "--check", str(script)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
