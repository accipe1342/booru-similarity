"""
Multi-booru reverse image search + tag search
(rule34 / gelbooru / safebooru / e621 / danbooru).

Pipeline: image --> WD14 SwinV2_v3 (1024-d embedding + predicted tags)
          --> FAISS cosine KNN over a prebuilt index --> nearest post ids
          --> resolve to images (+ original post links).

UI: three tabs (Image search / Tag search / Settings), live status messages,
in-UI error reporting, tag-overlap rerank, editable tags, post links, and
auto-deletion of downloaded result images.
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
    "rule34 (~10M)":     ("deepghs/anime_sites_indices", "model", "SwinV2_v3_rule34_9972271_4GB",   "rule34",   "bare"),
    "gelbooru (~10M)":   ("deepghs/anime_sites_indices", "model", "SwinV2_v3_gelbooru_10067238_4GB", "gelbooru", "bare"),
    "danbooru (~8M)":    ("deepghs/anime_sites_indices", "model", "SwinV2_v3_danbooru_8005009_4GB",  "danbooru", "bare"),
    "ALL-IN-ONE (~78M)": ("deepghs/anime_sites_indices", "model", "SwinV2_v3_AIO20250525_78281430_8G", None,    "prefixed"),
}
SITES = ["rule34", "gelbooru", "safebooru", "e621", "danbooru"]
DEFAULT_MODE = os.environ.get("RETRIEVAL_MODE", "mirror")
hf_client = get_hf_client()
_RATINGS = ["general", "sensitive", "questionable", "explicit"]
N_TAGS_USED = 20


# --------------------------------------------------------------------------- #
# Helpers
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
        out.append(f"{site}_{int(s)}" if (s.lstrip("-").isdigit() and site) else s)
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
    emb, _r, general, _c = wd14.get_wd14_tags(
        img, model_name="SwinV2_v3", fmt=("embedding", "rating", "general", "character"))
    ordered = sorted(general.items(), key=lambda kv: kv[1], reverse=True)
    label = {k: float(v) for k, v in ordered[:15]}
    pred = [k.replace(" ", "_") for k, _ in ordered[:N_TAGS_USED]]
    emb = np.expand_dims(np.asarray(emb, dtype=np.float32), 0)
    faiss.normalize_L2(emb)
    return emb, label, " ".join(pred), set(pred)


def _links_md(rows) -> str:
    if not rows:
        return ""
    lines = ["### Post links"]
    for site, pid, label in rows:
        url = booru_resolver.post_url(site, pid)
        lines.append(f"- [{label}]({url})" if url else f"- {label}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Handler: visual similarity search
# --------------------------------------------------------------------------- #
def run_similarity(img_input, index_name, mode, n_neighbours, accepted_ratings,
                   rerank, auto_clean, progress=gr.Progress()):
    if img_input is None:
        return [], {}, "", "_Upload an image first._", ""
    try:
        if auto_clean:
            purge_results()
        repo_id, repo_type, model_name, site, id_style = INDEXES[index_name]

        progress(0.05, desc="Embedding image (WD14)…")
        emb, label, seed_tags, pred_set = _predict(img_input)

        progress(0.3, desc="Loading index (first use downloads several GB)…")
        ids, index = _load_index(repo_id, repo_type, model_name)

        progress(0.55, desc="Searching nearest neighbours…")
        dists, idxs = index.search(emb, k=n_neighbours)
        raw_ids = _raw_ids(ids[idxs][0], site, id_style)
        accepted = set(accepted_ratings)
        sess = requests.Session()

        items = []
        if mode == "live":
            progress(0.7, desc="Fetching posts from booru API…")
            for rid, dist in zip(raw_ids, dists[0]):
                s, num = rid.rsplit("_", 1)
                m = booru_resolver.fetch_meta(s, int(num), session=sess)
                if not m or not m["url"] or not booru_resolver._passes(m["rating"], accepted):
                    continue
                items.append((rid, dist, m["url"], m["tags"]))
        else:
            progress(0.7, desc="Downloading images from HF mirror…")
            img_map = _fetch_mirror(raw_ids)
            tagmap = {}
            if rerank:
                progress(0.85, desc="Fetching tags for rerank…")
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

        progress(1.0, desc="Done")
        if not results:
            hint = ("No images returned. " + (
                "Mirror mode needs a valid HF token (Settings tab)."
                if mode == "mirror" else
                "Try enabling more ratings, or the booru API may be rate-limiting."))
            return [], label, seed_tags, f"⚠️ {hint}", ""
        return results, label, seed_tags, f"✅ {len(results)} results.", _links_md(link_rows)
    except Exception as e:  # noqa: BLE001
        return [], {}, "", f"❌ Error: {e}", ""


# --------------------------------------------------------------------------- #
# Handler: native tag search
# --------------------------------------------------------------------------- #
def run_tag_search(tags_text, site, n_neighbours, accepted_ratings, auto_clean,
                   progress=gr.Progress()):
    if not tags_text or not tags_text.strip():
        return [], "_Enter some tags first._"
    try:
        if auto_clean:
            purge_results()
        accepted = set(accepted_ratings)
        tags = [t for t in re.split(r"[,\s]+", tags_text.strip()) if t]
        cap = booru_resolver.TAG_SEARCH_LIMIT.get(site, 6)
        progress(0.3, desc=f"Searching {site} tags…")
        hits = booru_resolver.tag_search(site, tags[:cap], n_neighbours, accepted,
                                         session=requests.Session())
        progress(1.0, desc="Done")
        results, link_rows = [], []
        for h in hits:
            if not h["url"]:
                continue
            label = f"{site}_{h['id']}"
            results.append((h["url"], label))
            link_rows.append((site, h["id"], label))
        note = (f"Searched **{site}** for: `{' '.join(tags[:cap])}`"
                + ("" if len(tags) <= cap else f"  _(capped at {cap} tags)_"))
        body = _links_md(link_rows) if link_rows else \
            "_No results — tags may not match this site's vocabulary._"
        return results, note + "\n\n" + body
    except Exception as e:  # noqa: BLE001
        return [], f"❌ Error: {e}"


def apply_settings(token, persist, contact):
    global hf_client
    msgs = []
    if token and token.strip():
        tok = token.strip()
        try:
            if persist:
                from huggingface_hub import login
                login(token=tok, add_to_git_credential=False)
                msgs.append("HF token applied and saved to this machine.")
            else:
                os.environ["HF_TOKEN"] = tok
                msgs.append("HF token applied for this session.")
            hf_client = get_hf_client()
        except Exception as e:  # noqa: BLE001
            msgs.append(f"Token error: {e}")
    if contact and contact.strip():
        booru_resolver.USER_AGENT = (
            f"booru_image_similarity/1.0 (self-hosted; contact: {contact.strip()})")
        msgs.append("e621 contact (User-Agent) updated.")
    return "  \n".join(f"✅ {m}" for m in msgs) if msgs else "Nothing to apply."


def clear_image_tab():
    purge_results()
    return [], {}, "", "", ""


def clear_tag_tab():
    purge_results()
    return [], ""


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #
def build_ui():
    with gr.Blocks(title="Booru Similarity") as demo:
        gr.Markdown("## Multi-Booru Image Similarity & Tag Search")

        with gr.Tab("🔍 Image search"):
            with gr.Row():
                img_input = gr.Image(type="pil", label="Input")
                with gr.Column():
                    index_name = gr.Dropdown(choices=list(INDEXES), value=list(INDEXES)[0],
                                             label="Index / site")
                    mode = gr.Radio(choices=["mirror", "live"], value=DEFAULT_MODE,
                                    label="Retrieval mode",
                                    info="mirror = HF mirrors (needs HF token). "
                                         "live = booru API (original posts + ratings).")
                    n_img = gr.Slider(1, 50, value=20, step=1, label="# of images")
                    ratings_img = gr.CheckboxGroup(choices=_RATINGS,
                                                   value=["general", "sensitive"],
                                                   label="Allowed ratings (live mode)")
                    rerank = gr.Checkbox(value=False, label="Rerank by tag overlap")
                    auto_clean_img = gr.Checkbox(value=True,
                                                 label="Delete result images each search")
                    with gr.Row():
                        find_btn = gr.Button("Find similar", variant="primary")
                        clear_img_btn = gr.Button("Clear")
            status_img = gr.Markdown()
            gallery_img = gr.Gallery(label="Results", columns=[5])
            tags_label = gr.Label(label="Detected tags (confidence)", num_top_classes=15)
            links_img = gr.Markdown()

        with gr.Tab("🏷️ Tag search"):
            gr.Markdown("Tags auto-fill here after an image search. Edit them, pick a "
                        "site, and search. (WD14 predicts danbooru-vocab tags; rename "
                        "as needed for other sites. danbooru allows only 2 tags.)")
            tags_box = gr.Textbox(label="Tags", lines=3,
                                  placeholder="e.g. 1girl solo blue_hair smile")
            with gr.Row():
                site_dd = gr.Dropdown(choices=SITES, value="rule34", label="Site")
                n_tag = gr.Slider(1, 50, value=20, step=1, label="# of images")
            ratings_tag = gr.CheckboxGroup(choices=_RATINGS, value=["general", "sensitive"],
                                           label="Allowed ratings")
            auto_clean_tag = gr.Checkbox(value=True, label="Delete result images each search")
            with gr.Row():
                tag_btn = gr.Button("Search these tags", variant="primary")
                clear_tag_btn = gr.Button("Clear")
            status_tag = gr.Markdown()
            gallery_tag = gr.Gallery(label="Results", columns=[5])

        with gr.Tab("⚙️ Settings"):
            gr.Markdown(
                "Mirror mode needs a HuggingFace token (the booru mirrors are gated). "
                "Use a **read-only** token from https://huggingface.co/settings/tokens. "
                "'Remember' stores it locally via the standard HF login; otherwise it "
                "lasts only for this session. Never commit your token.")
            hf_token = gr.Textbox(label="HuggingFace token", type="password",
                                  placeholder="hf_...")
            remember = gr.Checkbox(value=False, label="Remember on this machine")
            contact = gr.Textbox(label="e621 contact email (API User-Agent)",
                                 placeholder="you@example.com")
            apply_btn = gr.Button("Apply settings", variant="primary")
            settings_status = gr.Markdown()
            apply_btn.click(apply_settings, inputs=[hf_token, remember, contact],
                            outputs=[settings_status])

        # wiring (inputs order == handler signature order)
        find_btn.click(
            run_similarity,
            inputs=[img_input, index_name, mode, n_img, ratings_img, rerank, auto_clean_img],
            outputs=[gallery_img, tags_label, tags_box, status_img, links_img],
        )
        clear_img_btn.click(clear_image_tab,
                            outputs=[gallery_img, tags_label, tags_box, status_img, links_img])
        tag_btn.click(
            run_tag_search,
            inputs=[tags_box, site_dd, n_tag, ratings_tag, auto_clean_tag],
            outputs=[gallery_tag, status_tag],
        )
        clear_tag_btn.click(clear_tag_tab, outputs=[gallery_tag, status_tag])
    return demo


if __name__ == "__main__":
    purge_results()
    build_ui().queue().launch()
