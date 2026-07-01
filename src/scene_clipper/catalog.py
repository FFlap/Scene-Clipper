from __future__ import annotations

import json, subprocess, tempfile
from dataclasses import asdict, dataclass, replace
from pathlib import Path
import numpy as np
from PIL import Image


def similarity_threshold(value):
    value=max(0,min(100,float(value)))
    if value == 0: return 1.01
    return round(0.95-(value/100)*0.30,4)


@dataclass(frozen=True)
class SceneItem:
    episode: str
    start: float
    end: float
    midpoint: float
    perceptual_hash: int
    color: tuple[int, int, int]
    thumbnail: str
    video: str = ""
    similar_scene_count: int = 1
    id: str = ""
    embedding: tuple[float, ...] = ()


def filter_and_group(items, min_duration=3.0, hash_distance=18, color_distance=70, semantic_similarity=0.74):
    eligible=[item for item in items if item.end-item.start>=min_duration]
    semantic_matrix=None
    if eligible and all(item.embedding for item in eligible):
        matrix=np.asarray([item.embedding for item in eligible],dtype=np.float32)
        matrix/=np.clip(np.linalg.norm(matrix,axis=1,keepdims=True),1e-8,None)
        semantic_matrix=matrix@matrix.T
    representatives=[]; representative_indices=[]
    for item_index,item in enumerate(eligible):
        match = None
        for index,(rep,rep_index) in enumerate(zip(representatives,representative_indices)):
            hash_delta = (item.perceptual_hash ^ rep.perceptual_hash).bit_count()
            color_delta = np.linalg.norm(np.asarray(item.color) - np.asarray(rep.color))
            semantic = semantic_matrix is not None and semantic_matrix[item_index,rep_index] >= semantic_similarity
            if (hash_delta <= hash_distance and color_delta <= color_distance) or semantic:
                match = index; break
        if match is None:
            representatives.append(item); representative_indices.append(item_index)
        else:
            rep = representatives[match]
            representatives[match] = replace(rep, similar_scene_count=rep.similar_scene_count + 1)
    return representatives


def _features(image):
    gray=np.asarray(image.convert("L").resize((9,8),Image.Resampling.LANCZOS))
    bits=(gray[:,1:]>gray[:,:-1]).flatten(); value=sum(int(v)<<i for i,v in enumerate(bits))
    color=tuple(int(x) for x in np.asarray(image.convert("RGB").resize((1,1)))[0,0])
    return value,color


class SiglipEmbedder:
    def __init__(self): self.processor=self.model=self.torch=None
    def embed(self, images):
        if self.model is None:
            import torch
            from transformers import AutoImageProcessor, SiglipVisionModel
            name="google/siglip-base-patch16-224"; self.processor=AutoImageProcessor.from_pretrained(name); self.model=SiglipVisionModel.from_pretrained(name).eval(); self.torch=torch
        outputs=[]
        for start in range(0,len(images),24):
            batch=self.processor(images=images[start:start+24],return_tensors="pt")
            with self.torch.inference_mode(): values=self.model(**batch).pooler_output.numpy()
            values/=np.linalg.norm(values,axis=1,keepdims=True); outputs.extend(values)
        return outputs


def preprocess_episode(video: Path, output_dir: Path, rebuild=False, embedder=None, id_prefix: str = ""):
    episode=video.stem; target=output_dir/episode; catalog=target/"catalog.json"
    if catalog.exists() and not rebuild:
        return load_catalog(catalog)
    from scenedetect import ContentDetector, SceneManager, open_video
    stream=open_video(str(video)); manager=SceneManager(); manager.add_detector(ContentDetector(threshold=27)); manager.detect_scenes(stream)
    scenes=manager.get_scene_list(start_in_scene=True); target.mkdir(parents=True,exist_ok=True)
    raw=[]
    with tempfile.TemporaryDirectory() as td:
        for index,(start,end) in enumerate(scenes,1):
            mid=(start.seconds+end.seconds)/2; path=Path(td)/f"{index:04}.png"
            subprocess.run(["ffmpeg","-hide_banner","-loglevel","error","-ss",str(mid),"-i",str(video),"-frames:v","1","-vf","scale=240:-2",str(path)],check=True)
            with Image.open(path) as loaded: image=loaded.convert("RGB"); ph,color=_features(image)
            scene_id=f"{episode}-{index:04}"
            if id_prefix:
                scene_id=f"{id_prefix}:{scene_id}"
            raw.append(SceneItem(episode,start.seconds,end.seconds,mid,ph,color,str(path),str(video),id=scene_id))
        eligible=[item for item in raw if item.end-item.start>=3.0]
        persisted=[]
        for number,item in enumerate(eligible,1):
            name=f"scene-{number:04}.jpg"; Image.open(item.thumbnail).save(target/name,quality=68,optimize=True)
            persisted.append(replace(item,thumbnail=str((target/name).resolve())))
        embedder=embedder or SiglipEmbedder(); embeddings=np.asarray(embedder.embed([Image.open(x.thumbnail).convert("RGB") for x in persisted]),dtype=np.float16)
        np.savez_compressed(target/"features.npz",embeddings=embeddings,hashes=np.asarray([x.perceptual_hash for x in persisted],dtype=np.uint64),colors=np.asarray([x.color for x in persisted],dtype=np.uint8))
        records=[]
        for item in persisted:
            record=asdict(item); record.pop("embedding",None); record.pop("perceptual_hash",None); record.pop("color",None); records.append(record)
    payload={"episode":episode,"video":str(video),"original_scene_count":len(scenes),"eligible_scene_count":len(records),"scenes":records}
    catalog.write_text(json.dumps(payload,indent=2)+"\n"); return load_catalog(catalog)


VIDEO_EXTENSIONS = {".mkv", ".mp4", ".mov", ".m4v", ".avi", ".webm"}


def find_videos(video_dir: Path):
    return sorted(
        path
        for path in Path(video_dir).rglob("*")
        if path.is_file()
        and path.suffix.lower() in VIDEO_EXTENSIONS
        and not path.name.startswith(".")
    )


def preprocess_directory(video_dir: Path, output_dir: Path, rebuild=False, *, continue_on_error=False, id_prefix: str = ""):
    embedder=SiglipEmbedder()
    results=[]; failures=[]
    for video in find_videos(video_dir):
        try:
            results.append(preprocess_episode(video,output_dir,rebuild,embedder,id_prefix=id_prefix))
        except Exception as exc:
            if not continue_on_error:
                raise
            failures.append({"video":str(video),"error":str(exc)})
    if continue_on_error:
        return {"results":results,"failures":failures}
    return results


def load_catalog(catalog_path: Path, slider=70):
    payload=json.loads(Path(catalog_path).read_text()); feature_path=Path(catalog_path).with_name("features.npz")
    if not feature_path.exists(): return payload
    features=np.load(feature_path); items=[]
    for index,(record,embedding) in enumerate(zip(payload["scenes"],features["embeddings"])):
        items.append(SceneItem(
            record["episode"],record["start"],record["end"],record["midpoint"],
            int(features["hashes"][index]),tuple(int(x) for x in features["colors"][index]),
            record["thumbnail"],record["video"],id=record["id"],embedding=tuple(float(x) for x in embedding),
        ))
    reps=filter_and_group(items,semantic_similarity=similarity_threshold(slider))
    result={**payload,"representative_count":len(reps),"similarity_slider":float(slider),"scenes":[]}
    for item in reps:
        record=asdict(item); record.pop("embedding",None); record.pop("perceptual_hash",None); record.pop("color",None); result["scenes"].append(record)
    return result
