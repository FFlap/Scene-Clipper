import base64
import io
import json
import urllib.error

import pytest
from PIL import Image

from scene_clipper.describe import (
    API_URL,
    DESCRIBE_SCHEMA,
    DESCRIPTION_FIELDS,
    MODEL,
    clip_describer,
    contact_sheet,
    describe_clip,
    frame_times,
    parse_description,
    post_json,
    sheet_base64,
)


def test_frame_times_samples_start_middle_and_end_inside_the_clip():
    times = frame_times(10, 14)

    assert times[0] == 10
    assert times[1] == 12
    assert 13.5 <= times[2] < 14


def test_contact_sheet_stitches_frames_into_one_wide_image():
    frames = [Image.new("RGB", (40, 30), colour) for colour in ("red", "green", "blue")]

    sheet = contact_sheet(frames, gap=10, label_height=20)

    assert sheet.width == 40 * 3 + 10 * 2
    assert sheet.height == 30 + 20


def test_describe_clip_posts_one_inline_image_to_the_google_model(monkeypatch, tmp_path):
    frames = [Image.new("RGB", (32, 18), "red") for _ in range(3)]
    sampled = []
    monkeypatch.setattr(
        "scene_clipper.describe.extract_frames",
        lambda video, times, **kwargs: sampled.append((video, times)) or frames,
    )
    sent = {}
    reply = {
        "summary": "A tense rooftop stand-off.", "action": "The camera pushes in as she turns.",
        "characters": [{"identity": "girl in red scarf", "gender": "female", "appearance": "black bob, amber eyes", "expression": "wary", "action": "turns to face the camera"}],
        "setting": "school rooftop at dusk", "mood": "tense", "themes": ["rivalry"],
        "visual_style": "warm backlight, medium shot", "tags": ["rooftop", "dusk", "stand-off"],
    }

    def poster(url, payload, headers, timeout=180):
        sent.update(url=url, payload=payload, headers=headers)
        return {"candidates": [{"content": {"parts": [{"text": json.dumps(reply)}]}}]}

    description = describe_clip(tmp_path / "e01.mkv", 4, 8, "AIzaSyExampleKey123", poster=poster)

    assert description == reply
    assert sampled[0][1] == frame_times(4, 8)
    assert sent["url"] == API_URL.format(model=MODEL)
    assert sent["headers"] == {"x-goog-api-key": "AIzaSyExampleKey123"}
    config = sent["payload"]["generationConfig"]
    assert config["responseMimeType"] == "application/json"
    assert config["responseSchema"] == DESCRIBE_SCHEMA
    parts = sent["payload"]["contents"][0]["parts"]
    images = [part for part in parts if "inline_data" in part]
    assert len(images) == 1
    assert images[0]["inline_data"]["mime_type"] == "image/jpeg"
    assert base64.b64decode(images[0]["inline_data"]["data"])[:2] == b"\xff\xd8"
    prompt = next(part["text"] for part in parts if "text" in part)
    for topic in ("START", "MIDDLE", "END", "gender", "mood", "tags", "JSON only"):
        assert topic in prompt
    assert "montage" not in prompt.lower() and "energy" not in prompt.lower()


def test_description_json_survives_fences_and_stray_prose():
    parsed = parse_description('Sure!\n```json\n{"summary": "A quiet walk home.", "tags": ["walk", " dusk "], "characters": [{"identity": "boy"}]}\n```')

    assert parsed["summary"] == "A quiet walk home."
    assert parsed["tags"] == ["walk", "dusk"]
    assert parsed["characters"] == [{"identity": "boy", "gender": "", "appearance": "", "expression": "", "action": ""}]
    assert parsed["mood"] == ""
    assert set(parsed) == set(DESCRIPTION_FIELDS)


def test_a_reply_that_is_not_json_is_an_error(monkeypatch):
    monkeypatch.setattr("scene_clipper.describe.extract_frames", lambda video, times, **kwargs: [Image.new("RGB", (32, 18), "red")])

    def poster(url, payload, headers, timeout=180):
        return {"candidates": [{"content": {"parts": [{"text": "The model rambled instead of answering."}]}}]}

    with pytest.raises(RuntimeError, match="did not return JSON"):
        describe_clip("e01.mkv", 0, 2, "AIzaSyExampleKey123", poster=poster)


def test_every_request_asks_for_the_schema(monkeypatch):
    monkeypatch.setattr("scene_clipper.describe.extract_frames", lambda video, times, **kwargs: [Image.new("RGB", (32, 18), "red")])
    payloads = []

    def poster(url, payload, headers, timeout=180):
        payloads.append(payload)
        return {"candidates": [{"content": {"parts": [{"text": '{"summary": "A quiet walk home."}'}]}}]}

    clip_describer("AIzaSyExampleKey123", poster=poster)({"video": "e01.mkv", "start": 0, "end": 2})

    assert len(payloads) == 1
    assert payloads[0]["generationConfig"]["responseSchema"] == DESCRIBE_SCHEMA


def test_api_errors_are_reported_with_their_message(monkeypatch):
    monkeypatch.setattr("scene_clipper.describe.extract_frames", lambda video, times, **kwargs: [Image.new("RGB", (32, 18), "red")])

    def poster(url, payload, headers, timeout=180):
        return {"error": {"message": "API key not valid"}}

    with pytest.raises(RuntimeError, match="API key not valid"):
        describe_clip("e01.mkv", 0, 2, "AIzaSyExampleKey123", poster=poster)


class _Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def test_rate_limited_requests_are_retried(monkeypatch):
    slept = []
    monkeypatch.setattr("scene_clipper.describe.time.sleep", slept.append)
    replies = [
        urllib.error.HTTPError("url", 429, "Too Many Requests", {}, io.BytesIO(b"{}")),
        _Response(b'{"candidates":[{"content":{"parts":[{"text":"late but fine"}]}}]}'),
    ]
    monkeypatch.setattr(
        "scene_clipper.describe.urllib.request.urlopen",
        lambda request, timeout=None: (_ for _ in ()).throw(replies.pop(0)) if isinstance(replies[0], Exception) else replies.pop(0),
    )

    payload = post_json("https://example.test", {}, {}, backoff=0.0)

    assert payload["candidates"][0]["content"]["parts"][0]["text"] == "late but fine"
    assert len(slept) == 1


def test_repeated_rate_limits_give_up_with_the_status(monkeypatch):
    monkeypatch.setattr("scene_clipper.describe.time.sleep", lambda seconds: None)
    monkeypatch.setattr(
        "scene_clipper.describe.urllib.request.urlopen",
        lambda request, timeout=None: (_ for _ in ()).throw(
            urllib.error.HTTPError("url", 429, "Too Many Requests", {}, io.BytesIO(b'{"error":"slow down"}'))
        ),
    )

    with pytest.raises(RuntimeError, match="429"):
        post_json("https://example.test", {}, {}, backoff=0.0)


def test_sheet_base64_is_a_decodable_jpeg_payload():
    data = sheet_base64(contact_sheet([Image.new("RGB", (20, 12), "blue")]))

    assert base64.b64decode(data)[:2] == b"\xff\xd8"
    assert json.dumps({"data": data})
