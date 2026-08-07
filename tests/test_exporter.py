import json
from pathlib import Path
from PIL import Image
import pytest

from scene_clipper.describe import DESCRIPTION_FIELDS
from scene_clipper.exporter import export_selection, ffmpeg_clip_command, format_filename_timestamp


def test_empty_selection_rejected(tmp_path):
    with pytest.raises(ValueError): export_selection([], tmp_path, False)


def test_grid_only_export_writes_grid_and_manifest(tmp_path):
    thumb = tmp_path / "thumb.jpg"; Image.new("RGB", (100, 60), "red").save(thumb)
    selected = [{"id":"x", "episode":"E01", "video":"/tmp/source episode.mkv", "start":1.0, "end":5.0, "thumbnail":str(thumb)}]
    out = export_selection(selected, tmp_path / "exports", False)
    assert (out / "selected-grid.jpg").exists()
    manifest = json.loads((out / "selection.json").read_text())
    assert manifest["clips_generated"] is False
    assert manifest["scenes"][0]["source_file"] == "source episode.mkv"
    assert manifest["scenes"][0]["timestamp"] == "0:01.000–0:05.000"
    assert not list(out.glob("*.mp4"))


def test_ffmpeg_command_is_frame_accurate_h264_aac():
    cmd = ffmpeg_clip_command("a.mkv", 1, 5, "x.mp4")
    assert "libx264" in cmd and "aac" in cmd
    assert cmd.index("-i") < cmd.index("-ss")


def test_ffmpeg_command_selects_audio_and_optionally_embeds_subtitles():
    cmd = ffmpeg_clip_command("a.mkv", 1, 5, "x.mp4", audio_track=2, subtitle_track=1)

    assert ["-map", "0:a:2"] == cmd[cmd.index("0:a:2") - 1:cmd.index("0:a:2") + 1]
    assert ["-map", "0:s:1"] == cmd[cmd.index("0:s:1") - 1:cmd.index("0:s:1") + 1]
    assert "mov_text" in cmd


def test_ffmpeg_command_omits_subtitles_when_disabled():
    cmd = ffmpeg_clip_command("a.mkv", 1, 5, "x.mp4", audio_track=0)

    assert not any(value.startswith("0:s:") for value in cmd)


def test_export_selection_uses_requested_audio_language(monkeypatch, tmp_path):
    thumb = tmp_path / "thumb.jpg"; Image.new("RGB", (100, 60), "red").save(thumb)
    selected = [{"id":"x", "episode":"E01", "video":"/tmp/source.mkv", "start":1.0, "end":5.0, "thumbnail":str(thumb)}]
    calls = []

    def fake_run(cmd, **kwargs):
        if cmd[0] == "ffprobe":
            return type("Result", (), {"stdout": '{"streams":[{"index":1,"tags":{"language":"jpn"}},{"index":2,"tags":{"language":"eng"}}]}'})()
        calls.append(cmd)
        return type("Result", (), {"stdout": ""})()

    monkeypatch.setattr("scene_clipper.exporter.subprocess.run", fake_run)

    export_selection(selected, tmp_path / "exports", True, "eng")

    assert "0:a:1" in calls[0]
    assert Path(calls[0][-1]).name == "001-E01-[1s-5s].mp4"


def test_export_writes_one_summary_json_for_the_described_clips(tmp_path):
    thumb = tmp_path / "thumb.jpg"; Image.new("RGB", (100, 60), "red").save(thumb)
    selected = [
        {"id":"a", "episode":"E01", "video":"/tmp/e01.mkv", "start":1.0, "end":5.0, "thumbnail":str(thumb)},
        {"id":"b", "episode":"E01", "video":"/tmp/e01.mkv", "start":9.0, "end":12.0, "thumbnail":str(thumb)},
    ]

    def describer(scene):
        return {
            "summary": f"summary for {scene['id']}", "action": "she turns to the camera",
            "characters": [{"identity": "girl in red scarf", "appearance": "black bob", "expression": "wary", "action": "turns"}],
            "setting": "rooftop at dusk", "mood": "tense", "themes": ["rivalry"],
            "visual_style": "warm backlight", "tags": ["rooftop", "dusk"],
        }

    out = export_selection(selected, tmp_path / "exports", False, describer=describer)

    clips = json.loads((out / "summary.json").read_text())["clips"]
    assert [clip["summary"] for clip in clips] == ["summary for a", "summary for b"]
    assert clips[0]["tags"] == ["rooftop", "dusk"]
    assert clips[1]["clip"] == "002-E01-[9s-12s].mp4"
    assert set(clips[0]) == {"clip", *DESCRIPTION_FIELDS}
    assert sorted(path.name for path in out.iterdir()) == ["selected-grid.jpg", "selection.json", "summary.json"]
    assert "description" not in json.loads((out / "selection.json").read_text())["scenes"][0]


def test_a_failed_description_does_not_abort_the_export(tmp_path):
    thumb = tmp_path / "thumb.jpg"; Image.new("RGB", (100, 60), "red").save(thumb)
    selected = [
        {"id":"a", "episode":"E01", "video":"/tmp/e01.mkv", "start":1.0, "end":5.0, "thumbnail":str(thumb)},
        {"id":"b", "episode":"E01", "video":"/tmp/e01.mkv", "start":9.0, "end":12.0, "thumbnail":str(thumb)},
    ]

    def describer(scene):
        if scene["id"] == "a":
            raise RuntimeError("Google API returned 429: rate limited")
        return {"summary": "second description"}

    out = export_selection(selected, tmp_path / "exports", False, describer=describer)

    clips = json.loads((out / "summary.json").read_text())["clips"]
    assert clips[0] == {"clip": "001-E01-[1s-5s].mp4", "error": "Google API returned 429: rate limited"}
    assert clips[1]["summary"] == "second description"
    assert "error" not in clips[1]
    assert (out / "selected-grid.jpg").exists()


def test_export_without_a_describer_writes_no_summary(tmp_path):
    thumb = tmp_path / "thumb.jpg"; Image.new("RGB", (100, 60), "red").save(thumb)
    selected = [{"id":"x", "episode":"E01", "video":"/tmp/e01.mkv", "start":1.0, "end":5.0, "thumbnail":str(thumb)}]

    out = export_selection(selected, tmp_path / "exports", False)

    assert not (out / "summary.json").exists()
    assert "description" not in json.loads((out / "selection.json").read_text())["scenes"][0]


def test_summary_names_match_the_generated_mp4_files(monkeypatch, tmp_path):
    thumb = tmp_path / "thumb.jpg"; Image.new("RGB", (100, 60), "red").save(thumb)
    selected = [{"id":"a", "episode":"E01", "video":"/tmp/e01.mkv", "start":1.0, "end":5.0, "thumbnail":str(thumb)}]
    rendered = []

    def fake_run(cmd, **kwargs):
        if cmd[0] != "ffprobe":
            rendered.append(Path(cmd[-1]).name)
        return type("Result", (), {"stdout": '{"streams":[]}'})()

    monkeypatch.setattr("scene_clipper.exporter.subprocess.run", fake_run)

    out = export_selection(selected, tmp_path / "exports", True, describer=lambda scene: {"summary": "ok"})

    assert json.loads((out / "summary.json").read_text())["clips"][0]["clip"] == rendered[0]


def test_filename_timestamp_includes_minutes_only_when_needed():
    assert format_filename_timestamp(5) == "5s"
    assert format_filename_timestamp(65) == "1m05s"
