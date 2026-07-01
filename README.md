# Scene Clipper

Turn episode folders into a fast local scene-review UI, select the shots you want, and export a gallery plus optional MP4 clips.

## What it does

* Splits episodes into scenes with PySceneDetect.
* Filters out very short scenes.
* Creates small representative thumbnails for quick review.
* Groups visually similar shots with perceptual hashes and SigLIP embeddings.
* Exports selected scenes as re-encoded MP4 clips by default.

## Install for development

```bash
git clone <your-repo-url> scene-clipper
cd scene-clipper
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
brew install ffmpeg   # if needed
```

Requires **Python 3.11+** and **FFmpeg** on your PATH. The first preprocess can take a while because the SigLIP model is downloaded by `transformers`.

## Run

```bash
python app.py
```

Open **http://127.0.0.1:5052/**

## Use

1. Click **Process Episodes**.
2. Select or paste a folder containing episode videos.
3. Click **Process Episodes** in the popup.
4. Pick the generated library from the landing screen.
5. Adjust **Similarity** at the top if duplicate grouping is too weak or too aggressive.
6. Select representative shots.
7. Click **Export selection**.

By default, export writes both:

* `selected-grid.jpg`
* re-encoded `.mp4` clips

Enable **Generate Gallery Only** to skip MP4 clip generation.

Each export also writes `selection.json` with the source filename and timestamp for every selected scene.

## Commands

```bash
python app.py
python app.py --port 5052
python app.py preprocess /path/to/episodes --output data/scene-catalog
python app.py preprocess /path/to/episodes --output data/scene-catalog --rebuild
python -m pytest -q
```