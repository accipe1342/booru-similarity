# Multi-Booru Image Similarity

A reverse-image-search ("find visually similar posts") tool for **rule34, gelbooru,
safebooru, e621, and danbooru** — the same architecture as
[SmilingWolf/danbooru2022_image_similarity](https://huggingface.co/spaces/SmilingWolf/danbooru2022_image_similarity),
generalized to multiple boorus.

## How it works

```
input image
  → WD14 SwinV2_v3 tagger, taking the 1024-d embedding (via dghs-imgutils)
  → L2-normalize  (cosine similarity = inner product on unit vectors)
  → FAISS KNN search over a prebuilt index
  → nearest post ids
  → resolve ids to actual images
```

Only two things are booru-specific: **which images are in the index**, and **how a
post id becomes a picture**. Everything else (model, FAISS, UI) is shared.

## The shortcut: deepghs already did most of the work

You do **not** need to scrape or embed millions of images. The
[deepghs](https://huggingface.co/deepghs) community publishes:

- **Prebuilt FAISS indices** in [`deepghs/anime_sites_indices`](https://huggingface.co/deepghs/anime_sites_indices):
  - `SwinV2_v3_rule34_9972271_4GB` — rule34, ~10.0M posts ✅
  - `SwinV2_v3_gelbooru_10067238_4GB` — gelbooru, ~10.1M ✅
  - `SwinV2_v3_danbooru_8005009_4GB` — danbooru, ~8.0M ✅
  - `SwinV2_v3_AIO20250525_78281430_8G` — **all-in-one, ~78M** mixed sites (ids prefixed `site_postid`) ✅
  - plus yandere, konachan, zerochan, anime_pictures, fancaps, iqdb
- **WebP image mirrors** (`deepghs/<site>-webp-4Mpixel`) so retrieval never hits the live site.
- **`embeddings_tools`** — the exact collect→repack→train→publish pipeline used to build the indices.

Each index folder is uniform: `ids.npy`, `knn.index`, `infos.json`
(dim **1024**, `index_param="nprobe=23,efSearch=46,ht=2048"`).

> A near-identical reference app already exists: [`deepghs/search_image_by_image`](https://huggingface.co/spaces/deepghs/search_image_by_image).
> This package is a self-hostable version of it with an added **live-resolver mode** and **rating filter**.

### Per-site status

| Site       | Prebuilt index?            | Image mirror? | What you do            |
|------------|----------------------------|---------------|------------------------|
| rule34     | ✅ in anime_sites_indices  | ✅            | nothing — works now    |
| gelbooru   | ✅                         | ✅            | nothing — works now    |
| danbooru   | ✅                         | ✅            | nothing — works now    |
| e621       | ❌ (embeddings exist)      | ✅ (`e621_newest-webp-4Mpixel`) | build via Route A |
| safebooru  | ❌ (no deepghs mirror)     | ❌            | scrape + embed, Route B |

## Files

- `app.py` — Gradio app. Dropdown of indices, KNN search, gallery. Two retrieval modes (env `RETRIEVAL_MODE`):
  - `mirror` (default): pull webp from deepghs HF mirrors via `cheesechaser`.
  - `live`: hit the booru API for the **original** full-size post URL + apply the rating filter.
- `resolver.py` — id → live post URL for each site (rule34/gelbooru/safebooru = Gelbooru-style API; e621 and danbooru have their own schemas). Includes rating normalization (`safe→general`) and the safebooru `file_url` fallback.
- `pools.py` — generic cheesechaser mirror pool for any `deepghs/<site>-webp-4Mpixel`.
- `build_index.py` — build an index for e621/safebooru (or refresh any site). Route A wraps deepghs `embeddings_tools` (fast); Route B embeds from scratch (slow, GPU).
- `test_pipeline.py` — offline tests: FAISS build/save/reload/query self-NN check + resolver parsers for all sites. **All passing.**

## Run

```bash
pip install -r requirements.txt
python app.py                       # mirror mode (default)
RETRIEVAL_MODE=live python app.py   # links to original posts + rating filter
```

First query downloads the chosen index from HuggingFace (rule34/gelbooru ≈ 2.7 GB,
AIO ≈ 8 GB), so the box needs that much disk + RAM. Embedding the query image needs
the WD14 model (auto-downloaded by dghs-imgutils).

## Add e621 (Route A, recommended)

```bash
git clone https://github.com/deepghs/embeddings_tools && cd embeddings_tools
pip install -r requirements.txt
python -m embeddings_tools.collect hf -r deepghs/e621_newest-webp-4Mpixel -o /data/raw/e621
python -m embeddings_tools.repack  localx -i e621:/data/raw/e621 -o /data/repacked/e621
python -m embeddings_tools.faiss   -i /data/repacked/e621 -r you/booru_indices
```

Then add to `INDEXES` in `app.py`:

```python
"e621 (yours)": ("you/booru_indices", "model", "SwinV2_v3_e621_xxx", "e621", "bare"),
```

## Add safebooru (Route B)

No deepghs mirror exists, so scrape ids+images yourself (safebooru dapi:
`https://safebooru.org/index.php?page=dapi&s=post&q=index&json=1`), save as
`<postid>.jpg` in a folder, then:

```bash
python build_index.py --images /data/safebooru_imgs --out indices/safebooru --big
```

## Notes / caveats

- These sites host adult content (e621/rule34/gelbooru); safebooru and danbooru's
  `general` tier are SFW. `live` mode's rating filter defaults to `general,sensitive`.
  Keep your deployment access-controlled and follow each site's API terms (rate
  limits; e621 **requires** a descriptive User-Agent — set yours in `resolver.py`).
- Index/embedding model must match: an index built with SwinV2_v3 must be queried
  with SwinV2_v3 embeddings (1024-d). Don't mix tagger versions.
- The prebuilt indices are snapshots (dates in their names); they won't contain
  posts newer than the snapshot. Rebuild periodically with `embeddings_tools`.

## Text & blended search (SigLIP)

deepghs's SigLIP model (`smilingwolf/siglip_swinv2_base`, repo `deepghs/siglip_beta`)
was trained to share the **same embedding space** as the WD14 SwinV2_v3 tagger, so
text queries search the *same* indices — no extra index needed.

In the **Image search** tab there's a "Text query" box:

- text only -> SigLIP text encoder finds matching posts ("sunset, two girls, red dress")
- image only -> WD14 visual embedding (current behaviour, also yields tags)
- both -> normalized average of image + text embeddings (blended search)

Works best on the danbooru index (the model speaks danbooru concepts). The SigLIP
model auto-downloads on first text search.

## GPU acceleration

The per-query bottleneck is the ONNX tagger/encoder. To run it on a GPU, install
the GPU ONNX runtime (see `requirements-gpu.txt`):

```bash
pip install -r requirements.txt
pip uninstall -y onnxruntime && pip install onnxruntime-gpu   # NVIDIA/CUDA
# Windows, any GPU: pip install onnxruntime-directml  (instead of onnxruntime-gpu)
```

Then (NVIDIA) launch with `ONNX_MODE=cuda`. The app prints the active ONNX
providers on startup, e.g. `available providers: ['CUDAExecutionProvider', ...]`,
so you can confirm the GPU is in use.

## CI

`.github/workflows/ci.yml` runs `test_pipeline.py` (the offline FAISS + resolver
tests) on every push/PR against Python 3.11 and 3.12.

## More boorus

The dropdown now also includes the other prebuilt deepghs indices: **yandere,
konachan, zerochan, anime_pictures** (plus the 78M all-in-one).

- **Mirror mode** works for all of them (dedicated cheesechaser pools).
- **Live mode + tag search** work for yandere & konachan too (Moebooru API — note
  their rating `s` means *safe*, mapped to `general`). zerochan & anime_pictures
  are mirror-only (nonstandard APIs); live mode skips them gracefully, but result
  thumbnails still link to the original post.

## API credentials (rule34 & gelbooru)

rule34 and gelbooru now **require API credentials** for live/tag search (mirror
mode is unaffected — it uses the HF mirrors). Get an `api_key` + `user_id` from
your account on each site (Account → API access), enter them in the **Settings**
tab, and click Apply. They're appended to every API call for that site. Without
them, live/tag search on rule34/gelbooru returns "Missing authentication".

## Media types (images / gifs / videos)

Booru posts can be static images, gifs, or videos (webm/mp4). The **Media types**
filter (Advanced, on both tabs) lets you include/exclude each. Videos are shown as
their preview-image thumbnail with a ▶ badge (gifs get 🎞), and clicking still
opens the post. Mirror mode only returns static images, so the filter mainly
applies to live and tag search.
