from __future__ import annotations

import base64
import io
import json
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

from PIL import Image, ImageDraw

MODEL = "gemma-4-31b-it"
API_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
FRAME_LABELS = ("START", "MIDDLE", "END")

DESCRIBE_SCHEMA = {
    "type": "OBJECT",
    "required": ["summary", "action", "characters", "setting", "mood", "themes", "visual_style", "tags"],
    "properties": {
        "summary": {"type": "STRING", "description": "One sentence describing the whole clip."},
        "action": {"type": "STRING", "description": "What happens across the clip, including camera or subject movement."},
        "characters": {
            "type": "ARRAY",
            "description": "One entry per visible character; empty when nobody is on screen.",
            "items": {
                "type": "OBJECT",
                "required": ["identity", "gender", "appearance", "expression", "action"],
                "properties": {
                    "identity": {"type": "STRING", "description": "Series and character name if clearly recognised, otherwise a short label such as 'girl in red scarf'."},
                    "gender": {"type": "STRING", "description": "Apparent gender: male, female, ambiguous, or non-human."},
                    "appearance": {"type": "STRING", "description": "Hair colour and style, eye colour, skin tone, clothing and its colours, build, apparent age."},
                    "expression": {"type": "STRING", "description": "Facial expression and body language, e.g. wide-eyed shock, gritted teeth, slumped shoulders."},
                    "action": {"type": "STRING", "description": "What this character does in the clip."},
                },
            },
        },
        "setting": {"type": "STRING", "description": "Location, interior or exterior, time of day, weather."},
        "mood": {"type": "STRING", "description": "Emotional tone, e.g. tense, tender, melancholic, comedic, eerie."},
        "themes": {"type": "ARRAY", "description": "Themes or subject matter present in the clip.", "items": {"type": "STRING"}},
        "visual_style": {"type": "STRING", "description": "Colour palette, lighting, shot scale, framing, animation qualities such as speed lines, sakuga movement, held still frame, impact frames."},
        "tags": {"type": "ARRAY", "description": "8-15 lowercase search keywords.", "items": {"type": "STRING"}},
    },
}

DESCRIBE_PROMPT = """You are cataloguing anime clips so they can be searched later.

The image is one strip of three frames sampled from a SINGLE short clip, in order: START (left), MIDDLE (centre), END (right). They are the same shot moving through time, not three different clips.

Describe that clip and reply with JSON only - no prose, no markdown fence - matching exactly this shape:

{"summary": string, "action": string, "characters": [{"identity": string, "gender": string, "appearance": string, "expression": string, "action": string}], "setting": string, "mood": string, "themes": [string], "visual_style": string, "tags": [string]}

Guidance:
- summary: one sentence covering the whole clip.
- action: what happens across the clip, including camera or subject movement implied by the change between frames.
- characters: one entry per visible character, empty list when nobody is on screen. Give a series and character name only when you clearly recognise them; otherwise use a short label such as "boy in blue school blazer". Fill gender with male, female, ambiguous or non-human; appearance with hair colour and style, eye colour, skin tone, clothing and its colours, build and apparent age; expression with the face and body language.
- setting: location, interior or exterior, time of day, weather.
- mood: the emotional tone, for example tense, tender, melancholic, comedic, eerie.
- themes: the subject matter, for example ["friendship", "rivalry"].
- visual_style: colour palette, lighting, shot scale (close-up, medium, wide), framing, and animation qualities such as speed lines, sakuga movement, held still frame, sparkles, impact frames.
- tags: 8 to 15 lowercase search keywords.

Write about it as one continuous clip; never mention frames, panels, or the strip itself."""

DESCRIPTION_FIELDS = ("summary", "action", "characters", "setting", "mood", "themes", "visual_style", "tags")
CHARACTER_FIELDS = ("identity", "gender", "appearance", "expression", "action")


def frame_times(start, end):
    """Start, middle and end sample points, nudged inside the clip bounds."""
    start = max(0.0, float(start))
    end = max(start, float(end))
    duration = end - start
    tail = min(0.25, duration / 4) if duration else 0.0
    return [start, start + duration / 2, max(start, end - tail)]


def extract_frames(video, times, width=480):
    frames = []
    with tempfile.TemporaryDirectory() as workdir:
        for index, moment in enumerate(times, 1):
            path = Path(workdir) / f"frame-{index}.png"
            subprocess.run(
                ["ffmpeg", "-hide_banner", "-loglevel", "error", "-ss", f"{moment:.3f}",
                 "-i", str(video), "-frames:v", "1", "-vf", f"scale={width}:-2", str(path)],
                check=True, capture_output=True, text=True,
            )
            if not path.exists():
                raise RuntimeError(f"ffmpeg produced no frame at {moment:.3f}s of {video}")
            with Image.open(path) as loaded:
                frames.append(loaded.convert("RGB"))
    return frames


def contact_sheet(frames, gap=10, background=(16, 17, 20), label_height=22):
    """Stitch the sampled frames into one labelled left-to-right strip."""
    if not frames:
        raise ValueError("No frames to stitch")
    height = max(frame.height for frame in frames)
    width = sum(frame.width for frame in frames) + gap * (len(frames) - 1)
    sheet = Image.new("RGB", (width, height + label_height), background)
    draw = ImageDraw.Draw(sheet)
    offset = 0
    for index, frame in enumerate(frames):
        sheet.paste(frame, (offset, label_height))
        label = FRAME_LABELS[index] if index < len(FRAME_LABELS) else f"FRAME {index + 1}"
        draw.text((offset + 6, 6), label, fill=(247, 241, 232))
        offset += frame.width + gap
    return sheet


def sheet_base64(sheet, quality=85):
    buffer = io.BytesIO()
    sheet.save(buffer, format="JPEG", quality=quality, optimize=True)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


RETRY_CODES = {429, 500, 502, 503, 504}


class GoogleApiError(RuntimeError):
    def __init__(self, message, status=None):
        super().__init__(message)
        self.status = status


def post_json(url, payload, headers, timeout=180, attempts=3, backoff=6.0):
    """POST JSON, retrying the rate limits a free Google API key hands out."""
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", **headers}, method="POST",
    )
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8") or "{}")
        except urllib.error.HTTPError as exc:
            if exc.code in RETRY_CODES and attempt < attempts:
                time.sleep(backoff * attempt)
                continue
            detail = (exc.read().decode("utf-8", "replace") or "").strip()[:400]
            raise GoogleApiError(f"Google API returned {exc.code}: {detail or exc.reason}", exc.code) from exc
        except urllib.error.URLError as exc:
            raise GoogleApiError(f"Could not reach the Google API: {exc.reason}") from exc


def _reply_text(payload):
    candidates = payload.get("candidates") or []
    if not candidates:
        error = payload.get("error") or {}
        raise GoogleApiError(error.get("message") or "Google API returned no candidates")
    parts = (candidates[0].get("content") or {}).get("parts") or []
    text = "".join(part.get("text", "") for part in parts if isinstance(part, dict)).strip()
    if not text:
        raise GoogleApiError("Google API returned an empty description")
    return text


def _description_payload(image_base64, prompt):
    return {
        "contents": [{"role": "user", "parts": [
            {"text": prompt},
            {"inline_data": {"mime_type": "image/jpeg", "data": image_base64}},
        ]}],
        "generationConfig": {"responseMimeType": "application/json", "responseSchema": DESCRIBE_SCHEMA},
    }


def parse_description(text):
    """Read the schema reply, allowing only for a code fence around the JSON."""
    body = text.strip()
    if body.startswith("```"):
        body = body.split("\n", 1)[-1].rsplit("```", 1)[0]
    start, end = body.find("{"), body.rfind("}")
    if start < 0 or end <= start:
        raise RuntimeError(f"Model did not return JSON: {text.strip()[:200]}")
    try:
        parsed = json.loads(body[start:end + 1])
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Model returned malformed JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise RuntimeError("Model returned JSON that is not a clip description object")
    return {field: _coerce(field, parsed.get(field)) for field in DESCRIPTION_FIELDS}


def _coerce(field, value):
    if field == "characters":
        entries = value if isinstance(value, list) else []
        return [
            {key: str(entry.get(key, "") or "").strip() for key in CHARACTER_FIELDS}
            for entry in entries if isinstance(entry, dict)
        ]
    if field in ("themes", "tags"):
        values = value if isinstance(value, list) else ([value] if isinstance(value, str) and value else [])
        return [str(item).strip() for item in values if str(item).strip()]
    return str(value or "").strip()


def request_description(image_base64, api_key, prompt=DESCRIBE_PROMPT, poster=post_json):
    if not api_key:
        raise ValueError("Add a Google API key in Settings first")
    reply = poster(API_URL.format(model=MODEL), _description_payload(image_base64, prompt), {"x-goog-api-key": api_key})
    return parse_description(_reply_text(reply))


def describe_clip(video, start, end, api_key, prompt=DESCRIBE_PROMPT, poster=post_json):
    """Sample start/middle/end frames of a clip and ask the model to describe it."""
    sheet = contact_sheet(extract_frames(video, frame_times(start, end)))
    return request_description(sheet_base64(sheet), api_key, prompt, poster)


def clip_describer(api_key, prompt=DESCRIBE_PROMPT, poster=post_json):
    """Build the per-scene callable the exporter uses."""
    def describe(scene):
        return describe_clip(scene["video"], scene["start"], scene["end"], api_key, prompt, poster)
    return describe
