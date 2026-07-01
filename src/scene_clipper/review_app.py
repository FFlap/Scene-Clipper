from __future__ import annotations

import hashlib
import json
import re
import subprocess
import threading
import time
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_file

from .catalog import find_videos, load_catalog, preprocess_directory
from .exporter import export_selection


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug[:72] or "library"


def _library_id(path: Path) -> str:
    digest = hashlib.sha1(str(path.resolve()).encode("utf-8")).hexdigest()[:8]
    return f"{_slug(path.name)}-{digest}"


def _read_meta(path: Path) -> dict:
    meta_path = path / "library.json"
    if not meta_path.exists():
        return {}
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _write_meta(path: Path, source: Path, name: str | None = None) -> None:
    path.mkdir(parents=True, exist_ok=True)
    payload = {
        "name": name or source.name or str(source),
        "source": str(source.resolve()),
        "updated_at": time.time(),
    }
    (path / "library.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def create_app(catalog_root: Path, export_root: Path, exporter=export_selection):
    app = Flask(__name__)
    catalog_root = Path(catalog_root)
    export_root = Path(export_root)
    jobs: dict[str, dict] = {}
    scene_cache = {"mtime": 0.0, "mapping": {}}

    def _direct_catalogs(root: Path):
        return sorted(root.glob("*/catalog.json"))

    def _has_direct_catalogs(root: Path) -> bool:
        return bool(_direct_catalogs(root))

    def libraries():
        catalog_root.mkdir(parents=True, exist_ok=True)
        found = []
        if _has_direct_catalogs(catalog_root):
            found.append(
                {
                    "id": "__root__",
                    "name": _read_meta(catalog_root).get("name") or catalog_root.name,
                    "path": str(catalog_root.resolve()),
                    "source": _read_meta(catalog_root).get("source", ""),
                    "episode_count": len(_direct_catalogs(catalog_root)),
                    "active": True,
                }
            )
        for child in sorted(p for p in catalog_root.iterdir() if p.is_dir()):
            catalog_count = len(_direct_catalogs(child))
            if catalog_count == 0:
                continue
            meta = _read_meta(child)
            found.append(
                {
                    "id": child.name,
                    "name": meta.get("name") or child.name,
                    "path": str(child.resolve()),
                    "source": meta.get("source", ""),
                    "episode_count": catalog_count,
                    "active": True,
                }
            )
        return found

    def library_path(library_id: str | None) -> Path:
        if not library_id or library_id == "__root__":
            return catalog_root
        candidate = (catalog_root / library_id).resolve()
        root = catalog_root.resolve()
        if root not in candidate.parents and candidate != root:
            raise ValueError("Invalid library")
        return candidate

    def catalogs(slider=70, library_id: str | None = None):
        root = library_path(library_id)
        return [load_catalog(p, slider) for p in _direct_catalogs(root)]

    def scene_map():
        catalog_paths = [*catalog_root.glob("*/catalog.json"), *catalog_root.glob("*/*/catalog.json")]
        latest_mtime = max((path.stat().st_mtime for path in catalog_paths), default=0.0)
        if scene_cache["mtime"] == latest_mtime:
            return scene_cache["mapping"]
        mapping = {}
        for path in catalog_paths:
            for scene in json.loads(path.read_text(encoding="utf-8")).get("scenes", []):
                mapping[scene["id"]] = scene
        scene_cache.update(mtime=latest_mtime, mapping=mapping)
        return mapping

    def run_preprocess(job_id: str, source: Path, target: Path, rebuild: bool, name: str | None):
        try:
            videos = find_videos(source)
            if not videos:
                raise ValueError(f"No video files found in {source}")
            jobs[job_id].update(status="running", video_count=len(videos), message=f"Processing {len(videos)} videos")
            _write_meta(target, source, name)
            payload = preprocess_directory(source, target, rebuild, continue_on_error=True, id_prefix=target.name)
            results = payload["results"]
            failures = payload["failures"]
            scene_cache.update(mtime=0.0, mapping={})
            jobs[job_id].update(
                status="done" if not failures else "done_with_errors",
                message=f"Prepared {len(results)} videos" if not failures else f"Prepared {len(results)} videos; {len(failures)} failed",
                library_id=target.name,
                episode_count=len(results),
                failures=failures,
            )
        except Exception as exc:
            jobs[job_id].update(status="error", message=str(exc))

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/api/libraries")
    def library_index():
        return jsonify(libraries())

    @app.post("/api/choose-folder")
    def choose_folder():
        script = 'POSIX path of (choose folder with prompt "Select a video folder to preprocess")'
        try:
            result = subprocess.run(
                ["osascript", "-e", script],
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as exc:
            message = (exc.stderr or exc.stdout or "Folder picker cancelled").strip()
            return jsonify(error=message), 400
        return jsonify(path=result.stdout.strip())

    @app.get("/api/catalog")
    def catalog():
        try:
            return jsonify(catalogs(request.args.get("similarity", 20), request.args.get("library")))
        except ValueError as exc:
            return jsonify(error=str(exc)), 400

    @app.get("/api/preprocess/status")
    def preprocess_status():
        return jsonify(sorted(jobs.values(), key=lambda item: item["created_at"], reverse=True))

    @app.post("/api/preprocess")
    def preprocess():
        body = request.get_json() or {}
        source = Path(str(body.get("path", "")).strip()).expanduser()
        if not source.is_dir():
            return jsonify(error=f"Not a folder: {source}"), 400
        library_id = _library_id(source)
        target = catalog_root / library_id
        job_id = hashlib.sha1(f"{source.resolve()}:{time.time()}".encode("utf-8")).hexdigest()[:12]
        jobs[job_id] = {
            "id": job_id,
            "status": "queued",
            "message": "Queued",
            "source": str(source.resolve()),
            "library_id": library_id,
            "created_at": time.time(),
        }
        thread = threading.Thread(
            target=run_preprocess,
            args=(job_id, source, target, bool(body.get("rebuild")), body.get("name")),
            daemon=True,
        )
        thread.start()
        return jsonify(jobs[job_id])

    @app.get("/thumbnail/<scene_id>")
    def thumbnail(scene_id):
        scene = scene_map().get(scene_id)
        if not scene:
            return jsonify(error="Unknown scene"), 404
        return send_file(scene["thumbnail"], max_age=86400, conditional=True)

    @app.post("/api/export")
    def export():
        body = request.get_json() or {}
        selected = body.get("selected", [])
        mapping = scene_map()
        scenes = [mapping[x] for x in selected if x in mapping]
        if not scenes:
            return jsonify(error="Select at least one scene"), 400
        try:
            out = exporter(scenes, export_root, bool(body.get("generate_mp4")))
        except Exception as exc:
            return jsonify(error=str(exc)), 500
        return jsonify(output=str(out), count=len(scenes))

    return app
