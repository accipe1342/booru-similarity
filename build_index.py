#!/usr/bin/env python3
"""
Build a FAISS similarity index for a booru that deepghs hasn't pre-indexed
(e.g. e621, safebooru), or to refresh an existing one.

Two routes:

ROUTE A -- reuse deepghs precomputed WD14 embeddings (FAST, recommended).
  deepghs ships SwinV2_v3 embeddings inside their *-webp-4Mpixel mirrors under
  subdir `embs/SwinV2_v3`. e621 has deepghs/e621_newest-webp-4Mpixel. Use their
  embeddings_tools to collect -> repack -> train -> publish:

    pip install -r requirements.txt
    git clone https://github.com/deepghs/embeddings_tools && cd embeddings_tools
    pip install -r requirements.txt

    # 1. collect raw embeddings from the mirror's embs/SwinV2_v3 subdir
    python -m embeddings_tools.collect hf \
        -r deepghs/e621_newest-webp-4Mpixel -o /data/raw/e621
    # 2. repack + shuffle, prefixing ids as e621_<id>
    python -m embeddings_tools.repack localx \
        -i e621:/data/raw/e621 -o /data/repacked/e621
    # 3. train + upload the faiss index (ids.npy/knn.index/infos.json)
    python -m embeddings_tools.faiss -i /data/repacked/e621 -r you/booru_indices

  Then add an entry to INDEXES in app.py pointing at you/booru_indices.

ROUTE B -- embed from scratch (SLOW, only if no mirror exists, e.g. safebooru).
  This script implements Route B: scrape post ids+images, embed with WD14
  SwinV2_v3, then build the SAME index recipe deepghs uses. For millions of
  images use a GPU and expect hours-to-days; below is the reference pipeline.
"""
import argparse
import json
import os

import faiss
import numpy as np


# deepghs's production recipe (from anime_sites_indices infos.json):
#   index_key  = OPQ256_1280,IVF16384_HNSW32,PQ256x8
#   index_param= nprobe=23,efSearch=46,ht=2048
# For <~1M vectors a lighter factory trains faster with similar recall.
DIM = 1024


def embed_images(image_paths, batch=32):
    """WD14 SwinV2_v3 embeddings for a list of image paths -> (N, 1024) float32."""
    from imgutils.tagging import wd14
    embs, ids = [], []
    for p in image_paths:
        pid = int(os.path.splitext(os.path.basename(p))[0])
        e = wd14.get_wd14_tags(p, model_name="SwinV2_v3", fmt="embedding")
        embs.append(e); ids.append(pid)
    return np.asarray(embs, dtype=np.float32), np.asarray(ids)


def build(embs, ids, out_dir, big=False):
    """Build, save, and describe a FAISS cosine index from embeddings: writes knn.index, ids.npy, and infos.json. Uses the deepghs OPQ/IVF/PQ recipe when big=True, else a lighter IVF/Flat factory."""
    os.makedirs(out_dir, exist_ok=True)
    embs = embs.astype(np.float32).copy()
    faiss.normalize_L2(embs)                      # cosine via inner product

    if big:  # match deepghs exactly (needs >~100k vectors to train well)
        factory = "OPQ256_1280,IVF16384_HNSW32,PQ256x8"
        param = "nprobe=23,efSearch=46,ht=2048"
    else:    # small/medium datasets
        nlist = max(64, int(4 * np.sqrt(len(embs))))
        factory = f"IVF{nlist},Flat"
        param = "nprobe=32"

    index = faiss.index_factory(DIM, factory, faiss.METRIC_INNER_PRODUCT)
    index.train(embs)
    index.add(embs)

    faiss.write_index(index, os.path.join(out_dir, "knn.index"))
    np.save(os.path.join(out_dir, "ids.npy"), ids)
    with open(os.path.join(out_dir, "infos.json"), "w") as f:
        json.dump({"index_key": factory, "index_param": param,
                   "nb vectors": int(len(embs)), "vectors dimension": DIM}, f, indent=2)
    print(f"built {factory} over {len(embs)} vectors -> {out_dir}")
    print("files: knn.index, ids.npy, infos.json  (drop-in for app.py INDEXES)")


def main():
    """CLI entry point (Route B): embed every image in --images with WD14 and build an index into --out."""
    ap = argparse.ArgumentParser(description="Build a booru FAISS index (Route B).")
    ap.add_argument("--images", required=True, help="dir of <postid>.jpg/png/webp")
    ap.add_argument("--out", required=True, help="output index dir")
    ap.add_argument("--big", action="store_true", help="use deepghs OPQ/IVF/PQ recipe")
    args = ap.parse_args()

    paths = [os.path.join(args.images, f) for f in os.listdir(args.images)
             if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))]
    if not paths:
        raise SystemExit("no images found")
    embs, ids = embed_images(paths)
    build(embs, ids, args.out, big=args.big)


if __name__ == "__main__":
    main()
