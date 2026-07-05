import json

from scene_clipper import subtitles
from scene_clipper.review_app import create_app

SRT = """1
00:00:01,000 --> 00:00:03,500
<i>I won't give up!</i>

2
00:01:10,250 --> 00:01:12,000
{\\an8}Never say that again.

3
00:01:13,000 --> 00:01:14,000

"""


def test_parse_srt_strips_tags_and_times():
    cues = subtitles.parse_srt(SRT)
    assert cues == [
        {"start": 1.0, "end": 3.5, "text": "I won't give up!"},
        {"start": 70.25, "end": 72.0, "text": "Never say that again."},
    ]


def _fake_extraction(tmp_path, monkeypatch, video, srt_text):
    cache = tmp_path / "subcache"

    def fake_extract(v, cache_dir):
        target = cache / subtitles._cache_key(v)
        target.mkdir(parents=True, exist_ok=True)
        (target / "track-00-eng.srt").write_text(srt_text)
        return {"video": str(v), "mtime": 0, "tracks": [{"track": 0, "language": "eng", "title": "", "codec": "subrip", "text_based": True, "srt": "track-00-eng.srt"}]}

    monkeypatch.setattr(subtitles, "extract_subtitles", fake_extract)
    return cache


def test_search_folder_matches_case_insensitive(tmp_path, monkeypatch):
    video = tmp_path / "ep1.mkv"
    video.write_bytes(b"x")
    cache = _fake_extraction(tmp_path, monkeypatch, video, SRT)
    payload = subtitles.search_folder(tmp_path, "GIVE UP", cache)
    assert payload["video_count"] == 1
    assert payload["languages"] == ["eng"]
    result = payload["results"][0]
    assert result["text"] == "I won't give up!"
    assert result["after"] == "Never say that again."
    assert result["start"] == 1.0
    assert subtitles.search_folder(tmp_path, "missing phrase", cache)["results"] == []


def test_search_route_validates_input(tmp_path):
    client = create_app(tmp_path / "catalog", tmp_path / "exports").test_client()
    assert client.get("/api/subtitles/search?folder=/nope&q=hello").status_code == 400
    assert client.get(f"/api/subtitles/search?folder={tmp_path}&q=a").status_code == 400


def test_clip_route_exports_range(tmp_path, monkeypatch):
    video = tmp_path / "ep1.mkv"
    video.write_bytes(b"x")
    commands = []

    def fake_run(cmd, **kwargs):
        commands.append(cmd)
        return None

    monkeypatch.setattr("scene_clipper.review_app.subprocess.run", fake_run)
    client = create_app(tmp_path / "catalog", tmp_path / "exports").test_client()

    bad = client.post("/api/clip", json={"video": str(video), "start": 5, "end": 4})
    assert bad.status_code == 400

    response = client.post("/api/clip", json={"video": str(video), "start": 4.5, "end": 9.25})
    assert response.status_code == 200
    output = json.loads(response.data)["output"]
    assert output.endswith(".mp4") and "ep1" in output
    assert str(video) in commands[0]
    missing = client.post("/api/clip", json={"video": str(tmp_path / "nope.mkv"), "start": 0, "end": 2})
    assert missing.status_code == 400
