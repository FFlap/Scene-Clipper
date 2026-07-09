from __future__ import annotations

import json, subprocess
from datetime import datetime
from pathlib import Path
from PIL import Image,ImageDraw


def ffmpeg_clip_command(video, start, end, output, audio_track=None, subtitle_track=None):
    audio_map = "0:a?" if audio_track is None else f"0:a:{audio_track}"
    duration = max(0.0, float(end) - float(start))
    duration_text = f"{duration:g}"
    command = ["ffmpeg","-y","-hide_banner","-loglevel","error","-i",str(video),"-ss",str(start),"-t",duration_text,"-map","0:v:0","-map",audio_map]
    if subtitle_track is not None:
        command.extend(["-map", f"0:s:{subtitle_track}"])
    command.extend(["-c:v","libx264","-crf","18","-preset","veryfast","-c:a","aac","-b:a","192k"])
    if subtitle_track is not None:
        command.extend(["-c:s", "mov_text"])
    command.append(str(output))
    return command


def audio_tracks(video):
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a",
         "-show_entries", "stream=index:stream_tags=language,title",
         "-of", "json", str(video)],
        check=True, capture_output=True, text=True,
    )
    tracks = []
    for order, stream in enumerate(json.loads(result.stdout or "{}").get("streams", [])):
        tags = stream.get("tags", {})
        tracks.append({"track": order, "stream_index": stream.get("index"),
                       "language": tags.get("language", "und"), "title": tags.get("title", "")})
    return tracks


def audio_track_for_language(video, language):
    if not language:
        return None
    wanted = str(language).lower()
    for track in audio_tracks(video):
        if str(track.get("language", "")).lower() == wanted:
            return track["track"]
    return None


def format_timestamp(seconds):
    milliseconds=round(float(seconds)*1000)
    minutes,ms=divmod(milliseconds,60000)
    secs,ms=divmod(ms,1000)
    return f"{minutes}:{secs:02}.{ms:03}"


def format_filename_timestamp(seconds):
    minutes, secs = divmod(round(float(seconds)), 60)
    return f"{minutes}m{secs:02}s" if minutes else f"{secs}s"


def selection_record(item):
    source=Path(item["video"]).name
    timestamp=f"{format_timestamp(item['start'])}–{format_timestamp(item['end'])}"
    return {**item,"source_file":source,"timestamp":timestamp}


def export_selection(selected, root: Path, generate_clips: bool, audio_language: str | None = None):
    if not selected: raise ValueError("Select at least one scene")
    out=root/(datetime.now().strftime("%Y%m%d-%H%M%S-%f")); out.mkdir(parents=True)
    records=[selection_record(item) for item in selected]
    cols=4; cw,ch=360,238; rows=(len(records)+cols-1)//cols; grid=Image.new("RGB",(cols*cw,rows*ch),(16,17,20))
    for index,item in enumerate(records,1):
        with Image.open(item["thumbnail"]) as loaded: image=loaded.convert("RGB"); image.thumbnail((320,180))
        tile=Image.new("RGB",(cw,ch),(21,23,28)); tile.paste(image,(20,44)); draw=ImageDraw.Draw(tile)
        draw.text((10,8),item["source_file"][:48],fill=(247,241,232))
        draw.text((10,24),item["timestamp"],fill=(159,183,255))
        grid.paste(tile,(((index-1)%cols)*cw,((index-1)//cols)*ch))
        if generate_clips:
            audio_track = audio_track_for_language(item["video"], audio_language)
            timestamp_range = f"{format_filename_timestamp(item['start'])}-{format_filename_timestamp(item['end'])}"
            clip_name = f"{index:03}-{item['episode']}-[{timestamp_range}].mp4"
            subprocess.run(ffmpeg_clip_command(item["video"], item["start"], item["end"], out/clip_name, audio_track), check=True)
    grid.save(out/"selected-grid.jpg",quality=90)
    (out/"selection.json").write_text(json.dumps({"clips_generated":generate_clips,"scenes":records},indent=2)+"\n")
    return out
