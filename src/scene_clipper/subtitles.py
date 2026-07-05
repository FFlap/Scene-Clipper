from __future__ import annotations

import hashlib, json, re, subprocess
from pathlib import Path

from .catalog import find_videos

TEXT_SUBTITLE_CODECS = {"subrip", "srt", "ass", "ssa", "mov_text", "webvtt", "text"}
TIMESTAMP = re.compile(r"(\d+):(\d\d):(\d\d)[,.](\d{1,3})")


def probe_subtitle_tracks(video: Path):
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "s",
         "-show_entries", "stream=index,codec_name:stream_tags=language,title",
         "-of", "json", str(video)],
        check=True, capture_output=True, text=True,
    )
    tracks = []
    for order, stream in enumerate(json.loads(result.stdout or "{}").get("streams", [])):
        tags = stream.get("tags", {})
        tracks.append({
            "track": order,
            "codec": stream.get("codec_name", ""),
            "language": tags.get("language", "und"),
            "title": tags.get("title", ""),
            "text_based": stream.get("codec_name", "") in TEXT_SUBTITLE_CODECS,
        })
    return tracks


def _cache_key(video: Path) -> str:
    return hashlib.sha1(str(video.resolve()).encode("utf-8")).hexdigest()[:12]


def extract_subtitles(video: Path, cache_dir: Path):
    """Extract all text-based subtitle tracks of a video to SRT files, cached by mtime."""
    video = Path(video)
    target = Path(cache_dir) / _cache_key(video)
    meta_path = target / "meta.json"
    mtime = video.stat().st_mtime
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("mtime") == mtime and all((target / t["srt"]).exists() for t in meta["tracks"]):
            return meta
    target.mkdir(parents=True, exist_ok=True)
    tracks = []
    for info in probe_subtitle_tracks(video):
        if not info["text_based"]:
            continue
        name = f"track-{info['track']:02}-{re.sub(r'[^a-z0-9]+', '', info['language'].lower()) or 'und'}.srt"
        out = target / name
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(video),
                 "-map", f"0:s:{info['track']}", "-c:s", "srt", str(out)],
                check=True, capture_output=True, text=True,
            )
        except subprocess.CalledProcessError:
            continue
        tracks.append({**info, "srt": name})
    meta = {"video": str(video.resolve()), "mtime": mtime, "tracks": tracks}
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return meta


def _to_seconds(match: re.Match) -> float:
    hours, minutes, seconds, ms = match.groups()
    return int(hours) * 3600 + int(minutes) * 60 + int(seconds) + int(ms.ljust(3, "0")) / 1000


def parse_srt(text: str):
    cues = []
    for block in re.split(r"\n\s*\n", text.replace("\r\n", "\n").strip()):
        lines = [line for line in block.split("\n") if line.strip()]
        timing_index = next((i for i, line in enumerate(lines) if "-->" in line), None)
        if timing_index is None:
            continue
        stamps = TIMESTAMP.findall(lines[timing_index])
        if len(stamps) < 2:
            continue
        start = _to_seconds(TIMESTAMP.search(lines[timing_index]))
        end_match = list(TIMESTAMP.finditer(lines[timing_index]))[1]
        end = _to_seconds(end_match)
        content = " ".join(lines[timing_index + 1:])
        content = re.sub(r"<[^>]+>|\{\\[^}]*\}", "", content).strip()
        if content:
            cues.append({"start": start, "end": end, "text": content})
    return cues


_cue_cache: dict[str, tuple[float, list]] = {}


def load_cues(srt_path: Path):
    srt_path = Path(srt_path)
    mtime = srt_path.stat().st_mtime
    cached = _cue_cache.get(str(srt_path))
    if cached and cached[0] == mtime:
        return cached[1]
    cues = parse_srt(srt_path.read_text(encoding="utf-8", errors="replace"))
    _cue_cache[str(srt_path)] = (mtime, cues)
    return cues


def search_folder(folder: Path, query: str, cache_dir: Path, language: str | None = None, limit: int = 300):
    """Search a phrase across all subtitle tracks of every video in a folder."""
    query_lower = query.lower().strip()
    results, languages, failures = [], set(), []
    videos = find_videos(folder)
    for video in videos:
        try:
            meta = extract_subtitles(video, cache_dir)
        except (subprocess.CalledProcessError, OSError) as exc:
            failures.append({"video": str(video), "error": str(exc)})
            continue
        srt_dir = Path(cache_dir) / _cache_key(video)
        for track in meta["tracks"]:
            languages.add(track["language"])
            if language and track["language"] != language:
                continue
            cues = load_cues(srt_dir / track["srt"])
            for index, cue in enumerate(cues):
                if query_lower in cue["text"].lower():
                    results.append({
                        "video": str(video),
                        "episode": video.stem,
                        "track": track["track"],
                        "language": track["language"],
                        "track_title": track["title"],
                        "start": cue["start"],
                        "end": cue["end"],
                        "text": cue["text"],
                        "before": cues[index - 1]["text"] if index else "",
                        "after": cues[index + 1]["text"] if index + 1 < len(cues) else "",
                    })
                    if len(results) >= limit:
                        break
            if len(results) >= limit:
                break
    return {
        "query": query,
        "results": results,
        "languages": sorted(languages),
        "video_count": len(videos),
        "truncated": len(results) >= limit,
        "failures": failures,
    }


def cues_in_range(video: Path, track: int, start: float, end: float, cache_dir: Path):
    meta = extract_subtitles(video, cache_dir)
    srt_dir = Path(cache_dir) / _cache_key(Path(video))
    match = next((t for t in meta["tracks"] if t["track"] == track), None)
    if match is None:
        return []
    cues = load_cues(srt_dir / match["srt"])
    return [cue for cue in cues if cue["end"] >= start and cue["start"] <= end]
