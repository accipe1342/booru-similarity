"""
Multi-booru reverse image search + tag search
(rule34 / gelbooru / safebooru / e621 / danbooru).

Pipeline: image --> WD14 SwinV2_v3 (1024-d embedding + predicted tags)
          --> FAISS cosine KNN over a prebuilt index --> nearest post ids
          --> resolve to images (+ original post links).

UI: tabs (Image search / Tag search / Settings), Soft theme, live status,
in-UI errors, interactive tag chips, and clickable thumbnails that open the
original post. Mirror thumbnails are inlined as data URIs (nothing extra on disk).
"""
import os
import warnings

# Silence a harmless Starlette deprecation warning surfaced by Gradio per request.
warnings.filterwarnings("ignore", message=r".*HTTP_422_UNPROCESSABLE_ENTITY.*")

RESULTS_TMP = os.path.abspath(
    os.environ.setdefault("GRADIO_TEMP_DIR", os.path.join(os.getcwd(), "_gradio_tmp")))
os.makedirs(RESULTS_TMP, exist_ok=True)

import atexit
import base64
import html as html_lib
import io
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

INDEXES = {
    "rule34 (~10M)":     ("deepghs/anime_sites_indices", "model", "SwinV2_v3_rule34_9972271_4GB",   "rule34",   "bare"),
    "gelbooru (~10M)":   ("deepghs/anime_sites_indices", "model", "SwinV2_v3_gelbooru_10067238_4GB", "gelbooru", "bare"),
    "danbooru (~8M)":    ("deepghs/anime_sites_indices", "model", "SwinV2_v3_danbooru_8005009_4GB",  "danbooru", "bare"),
    "yandere (~1M)":     ("deepghs/anime_sites_indices", "model", "SwinV2_v3_yandere_1083589_1GB",    "yandere",        "bare"),
    "konachan (~300k)":  ("deepghs/anime_sites_indices", "model", "SwinV2_v3_konachan_309751_128MB", "konachan",       "bare"),
    "zerochan (~3.8M)":  ("deepghs/anime_sites_indices", "model", "SwinV2_v3_zerochan_3842375_1GB",  "zerochan",       "bare"),
    "anime_pictures (~600k)": ("deepghs/anime_sites_indices", "model", "SwinV2_v3_anime_pictures_605865_1GB", "anime_pictures", "bare"),
    "ALL-IN-ONE (~78M)": ("deepghs/anime_sites_indices", "model", "SwinV2_v3_AIO20250525_78281430_8G", None,    "prefixed"),
}
SITES = ["rule34", "gelbooru", "safebooru", "e621", "danbooru", "yandere", "konachan"]
DEFAULT_MODE = os.environ.get("RETRIEVAL_MODE", "mirror")
_RATINGS = ["general", "sensitive", "questionable", "explicit"]
N_TAGS_USED = 20
EXPLICIT_SITES = {"rule34", "gelbooru", "e621"}


def _default_ratings(site):
    """Sensible rating defaults per site (explicit-heavy sites include all)."""
    if site in EXPLICIT_SITES:
        return ["general", "sensitive", "questionable", "explicit"]
    return ["general", "sensitive"]
# SigLIP shares the WD14 SwinV2_v3 embedding space -> same indices work for text.
SIGLIP_REPO = "deepghs/siglip_beta"
SIGLIP_MODEL = "smilingwolf/siglip_swinv2_base_2025_02_22_18h56m54s"
hf_client = get_hf_client()

CONFIG_PATH = os.path.join(os.path.expanduser("~"), ".booru_similarity.json")


def load_config() -> dict:
    try:
        with open(CONFIG_PATH) as f:
            return json.load(f)
    except Exception:  # noqa: BLE001
        return {}


def save_config(cfg: dict) -> bool:
    try:
        with open(CONFIG_PATH, "w") as f:
            json.dump(cfg, f)
        try:
            os.chmod(CONFIG_PATH, 0o600)  # owner-only (POSIX)
        except OSError:
            pass
        return True
    except Exception:  # noqa: BLE001
        return False


def _apply_saved() -> dict:
    cfg = load_config()
    if cfg.get("rule34_key") and cfg.get("rule34_uid"):
        booru_resolver.set_credentials("rule34", cfg["rule34_key"], cfg["rule34_uid"])
    if cfg.get("gelbooru_key") and cfg.get("gelbooru_uid"):
        booru_resolver.set_credentials("gelbooru", cfg["gelbooru_key"], cfg["gelbooru_uid"])
    if cfg.get("contact"):
        booru_resolver.USER_AGENT = (
            f"booru_image_similarity/1.0 (self-hosted; contact: {cfg['contact']})")
    return cfg


SAVED = _apply_saved()


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
        YandeWebpDataPool, KonachanWebpDataPool, ZerochanWebpDataPool,
        AnimePicturesWebpDataPool,
    )
    from hfutils.utils import TemporaryDirectory
    from pools import quick_webp_pool

    site_cls = {"danbooru": DanbooruNewestWebpDataPool,
                "gelbooru": GelbooruWebpDataPool, "rule34": Rule34WebpDataPool,
                "yandere": YandeWebpDataPool, "konachan": KonachanWebpDataPool,
                "zerochan": ZerochanWebpDataPool,
                "anime_pictures": AnimePicturesWebpDataPool}
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
    pred = [k.replace(" ", "_") for k, _ in ordered[:N_TAGS_USED]]
    conf = {k: round(float(v), 3) for k, v in ordered[:15]}
    emb = np.expand_dims(np.asarray(emb, dtype=np.float32), 0)
    faiss.normalize_L2(emb)
    return emb, pred, " ".join(pred), set(pred), conf


def _embed(img, text):
    """Return (emb[1,D], pred_tags, seed_tag_str, pred_set).

    Image only -> WD14 (also yields tags). Text given -> SigLIP text encoder
    (same embedding space). Both given -> normalized average (text+image blend).
    """
    text = (text or "").strip()
    if not text:
        return _predict(img)
    from imgutils.generic import siglip
    t = np.asarray(siglip.siglip_text_encode(
        text, repo_id=SIGLIP_REPO, model_name=SIGLIP_MODEL, fmt="embeddings"),
        dtype=np.float32).reshape(1, -1)
    faiss.normalize_L2(t)
    if img is not None:
        i = np.asarray(siglip.siglip_image_encode(
            img, repo_id=SIGLIP_REPO, model_name=SIGLIP_MODEL, fmt="embeddings"),
            dtype=np.float32).reshape(1, -1)
        faiss.normalize_L2(i)
        emb = (i + t) / 2.0
        faiss.normalize_L2(emb)
    else:
        emb = t
    return emb, [], "", set(), {}


def _log_providers():
    try:
        import onnxruntime as ort
        print("[onnx] available providers:", ort.get_available_providers(),
              "| ONNX_MODE=", os.environ.get("ONNX_MODE", "(unset -> CPU)"))
    except Exception as e:  # noqa: BLE001
        print("[onnx] provider check failed:", e)


def _pil_to_datauri(im: Image.Image, size: int = 256) -> str:
    t = im.convert("RGB").copy()
    t.thumbnail((size, size))
    buf = io.BytesIO(); t.save(buf, format="JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def _grid(cards) -> str:
    """cards: list of (img_src, post_url, caption) -> clickable thumbnail grid."""
    if not cards:
        return ""
    cells = []
    for src, post, cap in cards:
        if not (isinstance(src, str) and src.startswith(("http://", "https://", "data:"))):
            continue
        src = html_lib.escape(src, quote=True)
        cap = html_lib.escape(cap)
        href = f' href="{html_lib.escape(post, quote=True)}" target="_blank" rel="noopener"' if post else ""
        cells.append(
            f'<a{href} style="text-decoration:none;color:inherit">'
            f'<div style="width:180px;margin:6px;display:inline-block;vertical-align:top">'
            f'<img src="{src}" loading="lazy" '
            f'style="width:180px;height:180px;object-fit:cover;border-radius:10px;'
            f'box-shadow:0 1px 4px rgba(0,0,0,.3)">'
            f'<div style="font-size:11px;opacity:.75;margin-top:3px">{cap}</div>'
            f'</div></a>')
    return ('<div style="display:flex;flex-wrap:wrap;justify-content:flex-start">'
            + "".join(cells) + "</div>")


# --------------------------------------------------------------------------- #
# Handlers
# --------------------------------------------------------------------------- #
def run_similarity(img_input, text_query, index_name, mode, n_neighbours, accepted_ratings,
                   rerank, media_types, auto_clean, progress=gr.Progress()):
    blank_chips = gr.update(choices=[], value=[])
    blank_conf = {}
    if img_input is None and not (text_query or "").strip():
        return "", blank_chips, "", "_Upload an image or enter a text query._", blank_conf
    try:
        if auto_clean:
            purge_results()
        repo_id, repo_type, model_name, site, id_style = INDEXES[index_name]
        media_types = set(media_types or [])

        progress(0.05, desc="Embedding query…")
        emb, pred_list, seed_tags, pred_set, conf = _embed(img_input, text_query)

        progress(0.3, desc="Loading index (first use downloads several GB)…")
        ids, index = _load_index(repo_id, repo_type, model_name)

        progress(0.55, desc="Searching nearest neighbours…")
        dists, idxs = index.search(emb, k=n_neighbours)
        raw_ids = _raw_ids(ids[idxs][0], site, id_style)
        accepted = set(accepted_ratings)
        sess = requests.Session()

        items = []  # (rid, dist, media, tags, mtype)
        if mode == "live":
            progress(0.7, desc="Fetching posts from booru API…")
            for rid, dist in zip(raw_ids, dists[0]):
                s, num = rid.rsplit("_", 1)
                if s not in booru_resolver.SITES:
                    continue  # AIO may contain sites with no live API here
                m = booru_resolver.fetch_meta(s, int(num), session=sess)
                if not m or not m["url"] or not booru_resolver._passes(m["rating"], accepted):
                    continue
                if m["type"] not in media_types:
                    continue
                thumb = m["preview"] if (m["type"] == "video" and m["preview"]) else m["url"]
                items.append((rid, dist, thumb, m["tags"], m["type"]))
        else:
            if "image" not in media_types:
                progress(1.0, desc="Done")
                return ("", gr.update(choices=pred_list, value=[]), seed_tags,
                        "⚠️ Mirror mode only returns static images; enable 'image' in Media types.", conf)
            progress(0.7, desc="Downloading images from HF mirror…")
            img_map = _fetch_mirror(raw_ids)
            tagmap = {}
            if rerank:
                progress(0.85, desc="Fetching tags for rerank…")
                for rid in img_map:
                    s, num = rid.rsplit("_", 1)
                    if s not in booru_resolver.SITES:
                        tagmap[rid] = set(); continue
                    m = booru_resolver.fetch_meta(s, int(num), session=sess)
                    tagmap[rid] = m["tags"] if m else set()
            for rid, dist in zip(raw_ids, dists[0]):
                if rid in img_map:
                    items.append((rid, dist, img_map[rid], tagmap.get(rid, set()), "image"))

        if rerank:
            items.sort(key=lambda it: (-len(pred_set & it[3]), -it[1]))

        cards = []
        for rid, dist, media, tags, mtype in items:
            s, num = rid.rsplit("_", 1)
            ov = len(pred_set & tags)
            badge = " ▶" if mtype == "video" else (" 🎞" if mtype == "gif" else "")
            cap = f"{rid} / {dist:.2f}{badge}" + (f" | tags:{ov}" if rerank else "")
            src = media if isinstance(media, str) else _pil_to_datauri(media)
            cards.append((src, booru_resolver.post_url(s, num), cap))

        progress(1.0, desc="Done")
        chips = gr.update(choices=pred_list, value=[])
        if not cards:
            hint = ("Mirror mode needs a valid HF token (Settings tab)."
                    if mode == "mirror" else
                    "Try more ratings/media types, or the API may be rate-limiting.")
            return "", chips, seed_tags, f"⚠️ No images returned. {hint}", conf
        return _grid(cards), chips, seed_tags, f"✅ {len(cards)} results.", conf
    except Exception as e:  # noqa: BLE001
        return "", blank_chips, "", f"❌ Error: {booru_resolver.redact(e)}", blank_conf


def run_tag_search(tags_text, site, n_neighbours, accepted_ratings, media_types, auto_clean,
                   progress=gr.Progress()):
    if not tags_text or not tags_text.strip():
        return "", "_Enter some tags first._"
    try:
        if auto_clean:
            purge_results()
        accepted = set(accepted_ratings)
        media_types = set(media_types or [])
        tags = [t for t in re.split(r"[,\s]+", tags_text.strip()) if t]
        cap = booru_resolver.TAG_SEARCH_LIMIT.get(site, 6)
        progress(0.3, desc=f"Searching {site} tags…")
        hits = booru_resolver.tag_search(site, tags[:cap], n_neighbours, accepted,
                                         session=requests.Session())
        progress(1.0, desc="Done")
        cards = []
        for h in hits:
            if not h["url"] or h["type"] not in media_types:
                continue
            thumb = h["preview"] if (h["type"] == "video" and h["preview"]) else h["url"]
            badge = " ▶" if h["type"] == "video" else (" 🎞" if h["type"] == "gif" else "")
            cards.append((thumb, booru_resolver.post_url(site, h["id"]), f"{site}_{h['id']}{badge}"))
        note = (f"Searched **{site}** for: `{' '.join(tags[:cap])}`"
                + ("" if len(tags) <= cap else f"  _(capped at {cap} tags)_"))
        if not cards:
            tips = []
            if site in ("rule34", "gelbooru"):
                tips.append(f"{site} needs API credentials (Settings) and uses plural tags like `1girls`")
            if site in EXPLICIT_SITES and not ({"questionable", "explicit"} & accepted):
                tips.append("tick questionable/explicit — this site is mostly adult")
            tips.append("try fewer or different tags")
            return "", note + "\n\n**No results.** Tips: " + "; ".join(tips) + "."
        return _grid(cards), note + f"  ·  {len(cards)} results."
    except Exception as e:  # noqa: BLE001
        return "", f"❌ Error: {booru_resolver.redact(e)}"


def send_chips(selected):
    return " ".join(selected or [])


def apply_settings(token, persist, contact, gel_key, gel_uid, r34_key, r34_uid, remember_creds):
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
            msgs.append(f"Token error: {booru_resolver.redact(str(e).replace(tok, '***'))}")
    if contact and contact.strip():
        booru_resolver.USER_AGENT = (
            f"booru_image_similarity/1.0 (self-hosted; contact: {contact.strip()})")
        msgs.append("e621 contact (User-Agent) updated.")
    if (gel_key or "").strip() or (gel_uid or "").strip():
        booru_resolver.set_credentials("gelbooru", gel_key, gel_uid)
        msgs.append("Gelbooru credentials "
                    + ("set." if (gel_key or "").strip() and (gel_uid or "").strip()
                       else "cleared (need both api_key and user_id)."))
    if (r34_key or "").strip() or (r34_uid or "").strip():
        booru_resolver.set_credentials("rule34", r34_key, r34_uid)
        msgs.append("Rule34 credentials "
                    + ("set." if (r34_key or "").strip() and (r34_uid or "").strip()
                       else "cleared (need both api_key and user_id)."))
    if remember_creds:
        cfg = load_config()
        if (r34_key or "").strip() and (r34_uid or "").strip():
            cfg["rule34_key"], cfg["rule34_uid"] = r34_key.strip(), r34_uid.strip()
        if (gel_key or "").strip() and (gel_uid or "").strip():
            cfg["gelbooru_key"], cfg["gelbooru_uid"] = gel_key.strip(), gel_uid.strip()
        if (contact or "").strip():
            cfg["contact"] = contact.strip()
        msgs.append("Saved to this machine (~/.booru_similarity.json, plaintext)."
                    if save_config(cfg) else "Could not write the config file.")
    return "  \n".join(f"✅ {m}" for m in msgs) if msgs else "Nothing to apply."


def clear_image_tab():
    purge_results()
    return "", gr.update(choices=[], value=[]), "", "", "", {}


def clear_tag_tab():
    purge_results()
    return "", ""


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #
try:
    THEME = gr.themes.Soft(primary_hue="orange", neutral_hue="slate").set(
        block_title_background_fill="*neutral_100",
        block_title_background_fill_dark="*neutral_800",
        block_title_text_color="*neutral_700",
        block_title_text_color_dark="*neutral_200",
        block_label_background_fill="*neutral_100",
        block_label_background_fill_dark="*neutral_800",
        block_label_text_color="*neutral_700",
        block_label_text_color_dark="*neutral_200",
    )
except Exception:  # invalid theme token on some versions -> plain Soft
    THEME = gr.themes.Soft(primary_hue="orange", neutral_hue="slate")


def build_ui():
    with gr.Blocks(theme=THEME, title="Booru Similarity") as demo:
        gr.Markdown("# 🔎 Multi-Booru Image Similarity & Tag Search")
        gr.Markdown("Find visually similar posts across rule34 · gelbooru · danbooru "
                    "(and e621 · safebooru via your own indices). Click any result to "
                    "open its original post.")

        with gr.Tab("🔍 Image search"):
            with gr.Row(equal_height=True):
                with gr.Column():
                    img_input = gr.Image(type="pil", label="Input", height=360)
                    text_query = gr.Textbox(
                        label="Text query (optional; blends with image if both set)",
                        placeholder="e.g. sunset, two girls, red dress")
                with gr.Column():
                    with gr.Group():
                        index_name = gr.Dropdown(choices=list(INDEXES),
                                                 value=list(INDEXES)[0], label="Index / site")
                        mode = gr.Radio(choices=["mirror", "live"], value=DEFAULT_MODE,
                                        label="Retrieval mode",
                                        info="mirror = HF mirrors (needs HF token). "
                                             "live = booru API (original posts + ratings).")
                        n_img = gr.Slider(1, 50, value=20, step=1, label="# of images")
                        ratings_img = gr.CheckboxGroup(choices=_RATINGS,
                                                       value=["general", "sensitive"],
                                                       label="Allowed ratings (live mode)")
                        with gr.Accordion("Advanced", open=False):
                            rerank = gr.Checkbox(value=False, label="Rerank by tag overlap")
                            media_types_img = gr.CheckboxGroup(
                                choices=["image", "gif", "video"],
                                value=["image", "gif", "video"],
                                label="Media types (live mode; mirror is images only)")
                            auto_clean_img = gr.Checkbox(value=True,
                                                         label="Delete result images each search")
                    with gr.Row():
                        find_btn = gr.Button("Find similar", variant="primary", scale=2)
                        clear_img_btn = gr.Button("Clear", scale=1)
            with gr.Accordion("🏷️ Detected tags", open=True):
                gr.Markdown("Tick tags, then send them to the Tag search tab.")
                tag_chips = gr.CheckboxGroup(choices=[], label="Detected tags")
                send_btn = gr.Button("Use selected → Tag search")
            with gr.Accordion("Tag confidence", open=False):
                tag_conf = gr.Label(num_top_classes=15,
                                    label="Detected tags by probability")
            status_img = gr.Markdown()
            results_img = gr.HTML()

        with gr.Tab("🏷️ Tag search"):
            gr.Markdown("Tags auto-fill from your last image search. Edit them, pick a site, "
                        "and search. WD14 predicts **danbooru-vocabulary** tags — rename for "
                        "other sites. danbooru allows only 2 tags per search.")
            tags_box = gr.Textbox(label="Tags", lines=2,
                                  placeholder="e.g. 1girl solo blue_hair smile")
            with gr.Row():
                site_dd = gr.Dropdown(choices=SITES, value="rule34", label="Site")
                n_tag = gr.Slider(1, 50, value=20, step=1, label="# of images")
            ratings_tag = gr.CheckboxGroup(choices=_RATINGS, value=["general", "sensitive"],
                                           label="Allowed ratings")
            with gr.Accordion("Advanced", open=False):
                media_types_tag = gr.CheckboxGroup(
                    choices=["image", "gif", "video"], value=["image", "gif", "video"],
                    label="Media types")
                auto_clean_tag = gr.Checkbox(value=True, label="Delete result images each search")
            with gr.Row():
                tag_btn = gr.Button("Search these tags", variant="primary", scale=2)
                clear_tag_btn = gr.Button("Clear", scale=1)
            status_tag = gr.Markdown()
            results_tag = gr.HTML()

        with gr.Tab("⚙️ Settings"):
            gr.Markdown(
                "Mirror mode needs a HuggingFace token (the booru mirrors are gated). "
                "Use a **read-only** token from https://huggingface.co/settings/tokens. "
                "'Remember' stores it locally via the standard HF login; otherwise it lasts "
                "only this session. Never commit your token.")
            hf_token = gr.Textbox(label="HuggingFace token", type="password", placeholder="hf_...")
            remember = gr.Checkbox(value=False, label="Remember on this machine")
            contact = gr.Textbox(label="e621 contact email (API User-Agent)",
                                 value=SAVED.get("contact", ""), placeholder="you@example.com")
            gr.Markdown("Gelbooru live/tag search needs API credentials "
                        "(Gelbooru → Account → API Access Credentials).")
            gel_key = gr.Textbox(label="Gelbooru api_key", type="password",
                                 placeholder="saved" if SAVED.get("gelbooru_key") else "...")
            gel_uid = gr.Textbox(label="Gelbooru user_id",
                                 value=SAVED.get("gelbooru_uid", ""), placeholder="12345")
            gr.Markdown("Rule34 live/tag search also needs API credentials "
                        "(rule34.xxx → account → API access).")
            r34_key = gr.Textbox(label="Rule34 api_key", type="password",
                                 placeholder="saved" if SAVED.get("rule34_key") else "...")
            r34_uid = gr.Textbox(label="Rule34 user_id",
                                 value=SAVED.get("rule34_uid", ""), placeholder="12345")
            remember_creds = gr.Checkbox(
                value=bool(SAVED), label="Remember booru credentials on this machine")
            apply_btn = gr.Button("Apply settings", variant="primary")
            settings_status = gr.Markdown(
                "✅ Loaded saved booru credentials." if SAVED else "")
            apply_btn.click(apply_settings,
                            inputs=[hf_token, remember, contact, gel_key, gel_uid,
                                    r34_key, r34_uid, remember_creds],
                            outputs=[settings_status])

        gr.Markdown("---\n<sub>Self-hosted. Mostly NSFW sources — use responsibly and follow "
                    "each site's API terms. Indices & mirrors by deepghs.</sub>")

        # per-site rating defaults
        site_dd.change(lambda s: gr.update(value=_default_ratings(s)),
                       inputs=[site_dd], outputs=[ratings_tag])
        index_name.change(lambda n: gr.update(value=_default_ratings(INDEXES[n][3])),
                          inputs=[index_name], outputs=[ratings_img])

        find_btn.click(
            run_similarity,
            inputs=[img_input, text_query, index_name, mode, n_img, ratings_img, rerank, media_types_img, auto_clean_img],
            outputs=[results_img, tag_chips, tags_box, status_img, tag_conf],
            show_progress_on=[status_img],
        )
        clear_img_btn.click(clear_image_tab,
                            outputs=[results_img, tag_chips, tags_box, status_img, text_query, tag_conf])
        send_btn.click(send_chips, inputs=[tag_chips], outputs=[tags_box])
        tag_btn.click(
            run_tag_search,
            inputs=[tags_box, site_dd, n_tag, ratings_tag, media_types_tag, auto_clean_tag],
            outputs=[results_tag, status_tag],
            show_progress_on=[status_tag],
        )
        clear_tag_btn.click(clear_tag_tab, outputs=[results_tag, status_tag])
    return demo


if __name__ == "__main__":
    _log_providers()
    purge_results()
    build_ui().queue().launch()
