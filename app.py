"""
Multi-booru reverse image search + tag search
(rule34 / gelbooru / safebooru / e621 / danbooru).

Pipeline: image --> WD14 SwinV2_v3 (1024-d embedding + predicted tags)
          --> FAISS cosine KNN over a prebuilt index --> nearest post ids
          --> resolve to images (+ original post links).

Features:
  * Visual similarity search (mirror or live retrieval)
  * Detected tags shown with confidence, and pre-filled into an EDITABLE box
  * "Search these tags" button -> native booru tag search on the edited tags
  * Optional rerank of visual results by tag overlap with the query image
  * Clickable links to the original post for every result
  * Auto-delete of downloaded result images (privacy)
"""
import os

# Pin Gradio's file cache to a folder WE control (set BEFORE importing gradio).
RESULTS_TMP = os.path.abspath(
    os.environ.setdefault("GRADIO_TEMP_DIR", os.path.join(os.getcwd(), "_gradio_tmp")))
os.makedirs(RESULTS_TMP, exist_ok=True)

import atexit
import json
import re
import shutil
from collections import defaultdict
from typing import List, Dict

import faiss
import gradio as gr
import numpy as np
import requests
from PIL import Image
from imgutils.tagging import wd14
from imgutils.utils import ts_lru_cache
from hfutils.operate import get_hf_client

import resolver as booru_resolver

# friendly name -> (repo_id, repo_type, model_name, site, id_style)
INDEXES = {
    "rule34 (~10M)":       ("deepghs/anime_sites_indices", "model", "SwinV2_v3_rule34_9972271_4GB",   "rule34",   "bare"),
    "gelbooru (~10M)":     ("deepghs/anime_sites_indices", "model", "SwinV2_v3_gelbooru_10067238_4GB", "gelbooru", "bare"),
    "danbooru (~8M)":      ("deepghs/anime_sites_indices", "model", "SwinV2_v3_danbooru_8005009_4GB",  "danbooru", "bare"),
    "ALL-IN-ONE (~78M)":   ("deepghs/anime_sites_indices", "model", "SwinV2_v3_AIO20250525_78281430_8G", None,    "prefixed"),
}

DEFAULT_MODE = os.environ.get("RETRIEVAL_MODE", "mirror")
hf_client = get_hf_client()
_RATINGS = ["general", "sensitive", "questionable", "explicit"]
N_TAGS_USED = 20  # how many top predicted tags to seed the editable box / overlap


# --------------------------------------------------------------------------- #
# Cleanup of cached result images
# --------------------------------------------------------------------------- #
def purge_results() -> None:
    for name in os.listdir(RESULTS_TMP):
        p = os.path.join(RESULTS_TMP, name)
        try:
            shutil.rmtree(p) if os.path.isdir(p) else os.remove(p)
        except OSError:
            pass


atexit.register(purge_results)


@ts_lru_cache(maxsize=2)
def _load_index(repo_id, repo_type, model_name):
    ids = np.load(hf_client.hf_hub_download(
        repo_id=repo_id, repo_type=repo_type, filename=f"{model_name}/ids.npy"))
    index = faiss.read_index(hf_client.hf_hub_download(
        repo_id=repo_id, repo_type=repo_type, filename=f"{model_name}/knn.index"))
    cfg = json.loads(open(hf_client.hf_hub_download(
        repo_id=repo_id, repo_type=repo_type, filename=f"{model_name}/infos.json")).read())["index_param"]
    faiss.ParameterSpace().set_index_parameters(index, cfg)
    return ids, index


def _raw_ids(neighbour_ids, site, id_style) -> List[str]:
    out = []
    for x in neighbour_ids:
        s = str(x).strip()
        if s.lstrip("-").isdigit() and site:
            out.append(f"{site}_{int(s)}")
        else:
            out.append(s)
    return out


def _fetch_mirror(raw_ids: List[str]) -> Dict[str, Image.Image]:
    from cheesechaser.datapool import (
        DanbooruNewestWebpDataPool, GelbooruWebpDataPool, Rule34WebpDataPool,
    )
    from hfutils.utils import TemporaryDirectory
    from pools import quick_webp_pool

    site_cls = {"danbooru": DanbooruNewestWebpDataPool,
                "gelbooru": GelbooruWebpDataPool, "rule34": Rule34WebpDataPool}
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


def _predict(img):
    """Return (embedding[1,1024], label_dict_top15, seed_tag_string, pred_set)."""
    emb, _rating, general, _char = wd14.get_wd14_tags(
        img, model_name="SwinV2_v3", fmt=("embedding", "rating", "general", "character"))
    ordered = sorted(general.items(), key=lambda kv: kv[1], reverse=True)
    label = {k: float(v) for k, v in ordered[:15]}
    pred = [k.replace(" ", "_") for k, _ in ordered[:N_TAGS_USED]]
    emb = np.expand_dims(np.asarray(emb, dtype=np.float32), 0)
    faiss.normalize_L2(emb)
    return emb, label, " ".join(pred), set(pred)


def _links_md(rows) -> str:
    """rows: list of (site, post_id, label_str). Build a clickable markdown list."""
    if not rows:
        return ""
    lines = ["### Post links"]
    for site, pid, label in rows:
        url = booru_resolver.post_url(site, pid)
        lines.append(f"- [{label}]({url})" if url else f"- {label}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Handler 1: visual similarity search (also fills the tag box)
# --------------------------------------------------------------------------- #
def run_similarity(img_input, index_name, mode, n_neighbours, accepted_ratings,
                   rerank, auto_clean):
    if img_input is None:
        return [], {}, "", ""
    if auto_clean:
        purge_results()

    repo_id, repo_type, model_name, site, id_style = INDEXES[index_name]
    emb, label, seed_tags, pred_set = _predict(img_input)
    ids, index = _load_index(repo_id, repo_type, model_name)
    dists, idxs = index.search(emb, k=n_neighbours)
    raw_ids = _raw_ids(ids[idxs][0], site, id_style)
    accepted = set(accepted_ratings)
    sess = requests.Session()

    items = []  # (rid, dist, media, tags)
    if mode == "live":
        for rid, dist in zip(raw_ids, dists[0]):
            s, num = rid.rsplit("_", 1)
            m = booru_resolver.fetch_meta(s, int(num), session=sess)
            if not m or not m["url"] or not booru_resolver._passes(m["rating"], accepted):
                continue
            items.append((rid, dist, m["url"], m["tags"]))
    else:
        img_map = _fetch_mirror(raw_ids)
        tagmap = {}
        if rerank:
            for rid in img_map:
                s, num = rid.rsplit("_", 1)
                m = booru_resolver.fetch_meta(s, int(num), session=sess)
                tagmap[rid] = m["tags"] if m else set()
        for rid, dist in zip(raw_ids, dists[0]):
            if rid in img_map:
                items.append((rid, dist, img_map[rid], tagmap.get(rid, set())))

    if rerank:
        items.sort(key=lambda it: (-len(pred_set & it[3]), -it[1]))

    results, link_rows = [], []
    for rid, dist, media, tags in items:
        s, num = rid.rsplit("_", 1)
        ov = len(pred_set & tags)
        cap = f"{rid} / {dist:.2f}" + (f" | tags:{ov}" if rerank else "")
        results.append((media, cap))
        link_rows.append((s, num, cap))
    return results, label, seed_tags, _links_md(link_rows)


# --------------------------------------------------------------------------- #
# Handler 2: native tag search on the (edited) tag box
# --------------------------------------------------------------------------- #
def run_tag_search(tags_text, index_name, n_neighbours, accepted_ratings, auto_clean):
    if not tags_text or not tags_text.strip():
        return [], "Enter some tags first."
    if auto_clean:
        purge_results()
    site = INDEXES[index_name][3] or "danbooru"  # AIO has no single site -> danbooru
    accepted = set(accepted_ratings)
    tags = [t for t in re.split(r"[,\s]+", tags_text.strip()) if t]
    cap = booru_resolver.TAG_SEARCH_LIMIT.get(site, 6)
    hits = booru_resolver.tag_search(site, tags[:cap], n_neighbours, accepted,
                                     session=requests.Session())
    results, link_rows = [], []
    for h in hits:
        if not h["url"]:
            continue
        label = f"{site}_{h['id']}"
        results.append((h["url"], label))
        link_rows.append((site, h["id"], label))
    note = (f"Searched {site} for: {' '.join(tags[:cap])}"
            + ("" if len(tags) <= cap else f"  (capped at {cap} tags)"))
    md = note + "\n\n" + (_links_md(link_rows) if link_rows
                          else "_No results — tags may not match this site's vocabulary._")
    return results, md


def clear_all():
    purge_results()
    return [], {}, "", ""


def build_ui():
    with gr.Blocks() as demo:
        gr.Markdown("## Multi-Booru Image Similarity & Tag Search")
        with gr.Row():
            img_input = gr.Image(type="pil", label="Input")
            with gr.Column():
                index_name = gr.Dropdown(choices=list(INDEXES), value=list(INDEXES)[0],
                                         label="Index / site")
                mode = gr.Radio(choices=["mirror", "live"], value=DEFAULT_MODE,
                                label="Retrieval mode (visual search)",
                                info="mirror = deepghs HF mirrors (needs HF_TOKEN). "
                                     "live = booru API (original posts + rating filter).")
                n_neighbours = gr.Slider(1, 50, value=20, step=1, label="# of images")
                accepted = gr.CheckboxGroup(choices=_RATINGS, value=["general", "sensitive"],
                                            label="Allowed ratings (live / tag search)")
                rerank = gr.Checkbox(value=False, label="Rerank visual results by tag overlap")
                auto_clean = gr.Checkbox(value=True,
                                         label="Delete downloaded result images each search")
                tags_box = gr.Textbox(
                    label="Tags (auto-filled from the image — edit, then 'Search these tags')",
                    lines=3, placeholder="e.g. 1girl solo blue_hair smile")
                with gr.Row():
                    find_btn = gr.Button("Find similar (visual)", variant="primary")
                    tag_btn = gr.Button("Search these tags")
                    clear_btn = gr.Button("Clear")
        gallery = gr.Gallery(label="Results", columns=[5])
        tags_label = gr.Label(label="Detected tags (confidence)", num_top_classes=15)
        links_md = gr.Markdown()

        find_btn.click(
            run_similarity,
            inputs=[img_input, index_name, mode, n_neighbours, accepted, rerank, auto_clean],
            outputs=[gallery, tags_label, tags_box, links_md],
        )
        tag_btn.click(
            run_tag_search,
            inputs=[tags_box, index_name, n_neighbours, accepted, auto_clean],
            outputs=[gallery, links_md],
        )
        clear_btn.click(clear_all, outputs=[gallery, tags_label, tags_box, links_md])
    return demo


if __name__ == "__main__":
    purge_results()
    build_ui().queue().launch()
