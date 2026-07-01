from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from scene_clipper.catalog import preprocess_directory  # noqa: E402
from scene_clipper.review_app import create_app  # noqa: E402


def main() -> int:
    argv = sys.argv[1:]
    if argv and argv[0] not in {"-h", "--help"} and argv[0].startswith("-"):
        argv = ["run", *argv]
    parser = argparse.ArgumentParser(description="Scene Clipper")
    subparsers = parser.add_subparsers(dest="command")
    run_parser = subparsers.add_parser("run", help="Run the web app")
    run_parser.add_argument("--catalog", type=Path, default=ROOT / "data" / "scene-catalog")
    run_parser.add_argument("--exports", type=Path, default=ROOT / "output" / "exports")
    run_parser.add_argument("--port", type=int, default=5052)
    preprocess_parser = subparsers.add_parser("preprocess", help="Preprocess an episode folder")
    preprocess_parser.add_argument("video_directory", type=Path)
    preprocess_parser.add_argument("--output", type=Path, default=ROOT / "data" / "scene-catalog")
    preprocess_parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "preprocess":
        result = preprocess_directory(args.video_directory, args.output, args.rebuild, continue_on_error=True)
        print(f"Prepared {len(result['results'])} videos; {len(result['failures'])} failed")
        return 0 if not result["failures"] else 1
    create_app(args.catalog, args.exports).run(host="127.0.0.1", port=args.port, debug=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
