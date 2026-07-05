from __future__ import annotations

import json, subprocess
from datetime import datetime
from pathlib import Path
from PIL import Image,ImageDraw


def ffmpeg_clip_command(video, start, end, output, audio_track=None, subtitle_track=None):
    audio_map = "0:a?" if audio_track is None else f"0:a:{audio_track}"
    command = ["ffmpeg","-y","-hide_banner","-loglevel","error","-i",str(video),"-ss",str(start),"-to",str(end),"-map","0:v:0","-map",audio_map]
    if subtitle_track is not None:
        command.extend(["-map", f"0:s:{subtitle_track}"])
    command.extend(["-c:v","libx264","-crf","18","-preset","medium","-c:a","aac","-b:a","192k"])
    if subtitle_track is not None:
        command.extend(["-c:s", "mov_text"])
    command.append(str(output))
    return command


def format_timestamp(seconds):
    milliseconds=round(float(seconds)*1000)
    minutes,ms=divmod(milliseconds,60000)
    secs,ms=divmod(ms,1000)
    return f"{minutes}:{secs:02}.{ms:03}"


def selection_record(item):
    source=Path(item["video"]).name
    timestamp=f"{format_timestamp(item['start'])}–{format_timestamp(item['end'])}"
    return {**item,"source_file":source,"timestamp":timestamp}


def export_selection(selected, root: Path, generate_clips: bool):
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
            subprocess.run(ffmpeg_clip_command(item["video"],item["start"],item["end"],out/f"{index:03}-{item['episode']}.mp4"),check=True)
    grid.save(out/"selected-grid.jpg",quality=90)
    (out/"selection.json").write_text(json.dumps({"clips_generated":generate_clips,"scenes":records},indent=2)+"\n")
    return out
