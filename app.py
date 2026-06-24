"""
Multi-booru reverse image search (rule34 / gelbooru / safebooru / e621 / danbooru).

Architecture (same as SmilingWolf/danbooru2022_image_similarity, generalized):
  input image --> WD14 SwinV2_v3 embedding (1024-d) --> L2 normalize
              --> FAISS cosine KNN search over a prebuilt index
              --> nearest post ids --> resolve to images.

Two retrieval modes (now selectable in the UI):
  * "mirror" (default): download a webp copy from deepghs HF mirrors via
                cheesechaser. No live-site traffic; works for every indexed id.
  * "live"  : hit the booru's public API for the original full-size URL and
                filter by rating (uses resolver.py). Gives links to real posts.

Prebuilt indices live in deepghs/anime_sites_indices. e621/safebooru have no
prebuilt index there -- build your own with build_index.py and point INDEXES at it.
"""
import json
import os
from collections import defaultdict
from typing import List, Dict

import faiss
import gradio as gr
import numpy as np
from PIL import Image
from imgutils.tagging import wd14
from imgutils.utils import ts_lru_cache
from hfutils.operate import get_hf_client

import resolver as booru_resolver

# --------------------------------------------------------------------------- #
# Index catalog. friendly name -> (repo_id, repo_type, model_name, site, id_style)
#   id_style "bare"     -> ids are ints, belong to `site`
#   id_style "prefixed" -> ids look like "rule34_123", site parsed per-id (AIO)
# --------------------------------------------------------------------------- #
INDEXES = {
    "rule34 (~10M)":       ("deepghs/anime_sites_indices", "model", "SwinV2_v3_rule34_9972271_4GB",   "rule34",   "bare"),
    "gelbooru (~10M)":     ("deepghs/anime_sites_indices", "model", "SwinV2_v3_gelbooru_10067238_4GB", "gelbooru", "bare"),
    "danbooru (~8M)":      ("deepghs/anime_sites_indices", "model", "SwinV2_v3_danbooru_8005009_4GB",  "danbooru", "bare"),
    "ALL-IN-ONE (~78M)":   ("deepghs/anime_sites_indices", "model", "SwinV2_v3_AIO20250525_78281430_8G", None,    "prefixed"),
    # After building e621/safebooru indices with build_index.py, add e.g.:
    # "e621 (yours)":      ("you/booru_indices", "model", "SwinV2_v3_e621_xxx", "e621", "bare"),
    # "safebooru (yours)": ("you/booru_indices", "model", "SwinV2_v3_safebooru_xxx", "safebooru", "bare"),
}

# Env var now only sets the UI default; mode is chosen per-search in the app.
DEFAULT_MODE = os.environ.get("RETRIEVAL_MODE", "mirror")
hf_client = get_hf_client()

_RATINGS = ["general", "sensitive", "questionable", "explicit"]


@ts_lru_cache(maxsize=2)
def _load_index(repo_id: str, repo_type: str, model_name: str):
    ids = np.load(hf_client.hf_hub_download(
        repo_id=repo_id, repo_type=repo_type, filename=f"{model_name}/ids.npy"))
    index = faiss.read_index(hf_client.hf_hub_download(
        repo_id=repo_id, repo_type=repo_type, filename=f"{model_name}/knn.index"))
    cfg = json.loads(open(hf_client.hf_hub_download(
        repo_id=repo_id, repo_type=repo_type, filename=f"{model_name}/infos.json")).read())["index_param"]
    faiss.ParameterSpace().set_index_parameters(index, cfg)
    return ids, index


def _raw_ids(neighbour_ids, site, id_style) -> List[str]:
    """Normalize FAISS-returned ids into 'site_postid' strings.

    deepghs indices may store ids already prefixed ('rule34_11348807') even for
    single-site indices, or as bare ints. Handle both: a pure-integer id with a
    known site gets the prefix added; anything already containing a prefix is
    kept as-is.
    """
    out = []
    for x in neighbour_ids:
        s = str(x).strip()
        if s.lstrip("-").isdigit() and site:   # bare int + known site
            out.append(f"{site}_{int(s)}")
        else:                                   # already 'site_postid'
            out.append(s)
    return out


# ---- mirror retrieval (deepghs HF webp mirrors via cheesechaser) ----------- #
def _fetch_mirror(raw_ids: List[str]) -> Dict[str, Image.Image]:
    from cheesechaser.datapool import (
        DanbooruNewestWebpDataPool, GelbooruWebpDataPool, Rule34WebpDataPool,
    )
    from hfutils.utils import TemporaryDirectory
    from pools import quick_webp_pool

    site_cls = {
        "danbooru": DanbooruNewestWebpDataPool,
        "gelbooru": GelbooruWebpDataPool,
        "rule34": Rule34WebpDataPool,
    }
    by_site = defaultdict(list)
    for rid in raw_ids:
        site, num = rid.rsplit("_", 1)
        by_site[site].append(int(num))

    out: Dict[str, Image.Image] = {}
    for site, nums in by_site.items():
        cls = site_cls.get(site) or quick_webp_pool(site, 3)
        with TemporaryDirectory() as td:
            cls().batch_download_to_directory(resource_ids=nums, dst_dir=td)
            for f in os.listdir(td):
                pid = int(os.path.splitext(f)[0])
                im = Image.open(os.path.join(td, f)); im.load()
                out[f"{site}_{pid}"] = im
    return out


# ---- live retrieval (original posts via resolver.py) ---------------------- #
def _fetch_live(raw_ids: List[str], accepted) -> Dict[str, str]:
    import requests
    sess = requests.Session()
    out: Dict[str, str] = {}
    for rid in raw_ids:
        site, num = rid.rsplit("_", 1)
        try:
            url = booru_resolver.resolve(site, int(num), set(accepted), session=sess)
        except ValueError:
            url = None
        if url:
            out[rid] = url
    return out


def search(img_input, index_name, mode, n_neighbours, accepted_ratings):
    if img_input is None:
        return []
    repo_id, repo_type, model_name, site, id_style = INDEXES[index_name]
    ids, index = _load_index(repo_id, repo_type, model_name)

    emb = wd14.get_wd14_tags(img_input, model_name="SwinV2_v3", fmt="embedding")
    emb = np.expand_dims(emb, 0).astype(np.float32)
    faiss.normalize_L2(emb)

    dists, idxs = index.search(emb, k=n_neighbours)
    neighbours = ids[idxs][0]
    raw_ids = _raw_ids(neighbours, site, id_style)

    results = []
    if mode == "live":
        url_map = _fetch_live(raw_ids, accepted_ratings)
        for rid, dist in zip(raw_ids, dists[0]):
            if rid in url_map:
                results.append((url_map[rid], f"{rid} / {dist:.2f}"))
    else:  # mirror
        img_map = _fetch_mirror(raw_ids)
        for rid, dist in zip(raw_ids, dists[0]):
            if rid in img_map:
                results.append((img_map[rid], f"{rid} / {dist:.2f}"))
    return results


def build_ui():
    with gr.Blocks() as demo:
        gr.Markdown("## Multi-Booru Image Similarity\n"
                    "rule34 / gelbooru / danbooru (prebuilt) + your own e621 / safebooru indices.")
        with gr.Row():
            img_input = gr.Image(type="pil", label="Input")
            with gr.Column():
                index_name = gr.Dropdown(choices=list(INDEXES), value=list(INDEXES)[0],
                                         label="Index")
                mode = gr.Radio(choices=["mirror", "live"], value=DEFAULT_MODE,
                                label="Retrieval mode",
                                info="mirror = images from deepghs HF mirrors (no booru traffic). "
                                     "live = original post URLs from the booru API + rating filter.")
                n_neighbours = gr.Slider(1, 50, value=20, step=1, label="# of images")
                accepted = gr.CheckboxGroup(choices=_RATINGS, value=["general", "sensitive"],
                                            label="Allowed ratings (applies to live mode only)")
                find_btn = gr.Button("Find similar images", variant="primary")
        gallery = gr.Gallery(label="Similar images", columns=[5])

        # inputs order == component creation order == search() signature order
        find_btn.click(
            search,
            inputs=[img_input, index_name, mode, n_neighbours, accepted],
            outputs=[gallery],
        )
    return demo


if __name__ == "__main__":
    build_ui().queue().launch()
