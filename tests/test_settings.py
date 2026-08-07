import json

from scene_clipper.describe import MODEL
from scene_clipper.settings import load_settings, public_settings, save_settings


def test_defaults_to_no_key(tmp_path):
    assert load_settings(tmp_path / "missing.json") == {"google_api_key": ""}


def test_save_writes_the_key(tmp_path):
    path = tmp_path / ".settings.json"

    settings = save_settings(path, {"google_api_key": "  AIzaSyExampleKey123  "})

    assert settings == {"google_api_key": "AIzaSyExampleKey123"}
    assert json.loads(path.read_text())["google_api_key"] == "AIzaSyExampleKey123"


def test_save_without_updates_keeps_the_stored_key(tmp_path):
    path = tmp_path / ".settings.json"
    save_settings(path, {"google_api_key": "AIzaSyExampleKey123"})

    assert save_settings(path, {})["google_api_key"] == "AIzaSyExampleKey123"


def test_corrupt_settings_file_falls_back_to_defaults(tmp_path):
    path = tmp_path / ".settings.json"
    path.write_text("{not json")

    assert load_settings(path) == {"google_api_key": ""}


def test_public_settings_hide_the_api_key():
    public = public_settings({"google_api_key": "AIzaSyExampleKey123"})

    assert "google_api_key" not in public
    assert public["has_api_key"] is True
    assert public["model"] == MODEL
    assert "ExampleKey" not in public["api_key_hint"]
    assert public["api_key_hint"].startswith("AIzaSy")
