import json
from pathlib import Path
from PIL import Image
import pytest

from scene_clipper.exporter import export_selection, ffmpeg_clip_command


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
