# Multi-Booru Image Similarity & Tag Search

[![CI](https://github.com/accipe1342/booru-similarity/actions/workflows/ci.yml/badge.svg)](https://github.com/accipe1342/booru-similarity/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

Reverse-image search and tag search across multiple booru sites — **rule34, gelbooru,
danbooru, e621, safebooru, yandere, konachan, zerochan, anime_pictures** — powered by
the WD14 tagger, deepghs's prebuilt FAISS indices, and a SigLIP text encoder.

Drop in an image to find visually similar posts, search by a text description, blend
both, or run a deliberate tag search — and click any result to open its original post.

> Generalizes [SmilingWolf/danbooru2022_image_similarity](https://huggingface.co/spaces/SmilingWolf/danbooru2022_image_similarity)
> to many boorus, using [deepghs](https://huggingface.co/deepghs) indices, mirrors, and models.

---

## Features

- **Visual similarity search** over ~8–78M images per index (FAISS cosine KNN).
- **Text & blended search** via deepghs's SigLIP model, which shares the WD14
  embedding space — so text queries hit the *same* indices, no extra index needed.
- **Native tag search** against each site's API, with tag-overlap awareness.
- **Two retrieval modes:** `mirror` (images from deepghs HF mirrors, no live-site
  traffic) and `live` (original post URLs from the booru API + rating filter).
- **Detected tags** with a confidence dropdown and interactive chips that feed tag search.
- **Clickable result thumbnails** that open the original post; videos shown as a
  preview thumbnail with a play badge, gifs flagged.
- **Per-site rating defaults**, media-type filtering (images / gifs / videos), and
  polite per-site rate limiting (throttle + retry/backoff).
- **Privacy:** result images auto-delete each search; credentials are redacted from
  all error output and never committed.

## How it works

```
query image -> WD14 SwinV2_v3      -> 1024-d embedding ----+
query text  -> SigLIP (same space) ----------------------- |
                                                           v
                                      L2-normalize -> FAISS cosine KNN
                                                           v
                                          nearest post ids (site_postid)
                                                           v
                    mirror: webp from HF mirror   |   live: booru API -> file_url
                                                           v
                            clickable thumbnail grid (+ links to posts)
```

Only two things are booru-specific: which images are in the index, and how a post id
becomes an image. Everything else (model, FAISS, UI) is shared.

## Supported sites

| Site             | Visual index | Mirror mode | Live / tag search         |
|------------------|--------------|-------------|---------------------------|
| rule34           | prebuilt     | yes         | yes (API key required)    |
| gelbooru         | prebuilt     | yes         | yes (API key required)    |
| danbooru         | prebuilt     | yes         | yes                       |
| yandere          | prebuilt     | yes         | yes (Moebooru API)        |
| konachan         | prebuilt     | yes         | yes (Moebooru API)        |
| zerochan         | prebuilt     | yes         | mirror only               |
| anime_pictures   | prebuilt     | yes         | mirror only               |
| ALL-IN-ONE (~78M)| prebuilt     | yes         | per-result where supported|
| e621 / safebooru | build your own (see below) | depends | e621 supported |

## Quick start

```bash
git clone https://github.com/accipe1342/booru-similarity.git
cd booru-similarity
python -m venv .venv
# Windows PowerShell: .\.venv\Scripts\Activate.ps1   |   bash: . .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Open the printed local URL. The first search downloads the chosen index from
HuggingFace (rule34/gelbooru ~2.7 GB, ALL-IN-ONE ~8 GB) and caches it, so the first
run is slow and needs that much disk + RAM.

**Mirror mode needs a HuggingFace token** (the booru mirrors are gated): create a
read-only token at <https://huggingface.co/settings/tokens> and paste it in Settings.

## Search modes

- **Image** — drop an image; WD14 embeds it and finds visually similar posts.
- **Text** — type a description (e.g. `sunset, red dress`); SigLIP searches the same index.
- **Blended** — provide both; the normalized average of image + text embeddings is searched.
- **Tag** — deliberate tag query against the site's API (use the chips to send detected tags).

## API credentials (rule34 & gelbooru)

rule34 and gelbooru require API credentials for live/tag search (mirror mode is
unaffected). Get an `api_key` + `user_id` from each site (Account -> API access),
enter them in the **Settings** tab, and tick "Remember booru credentials on this
machine" to persist them. They're stored in `~/.booru_similarity.json` (plaintext,
owner-only), kept out of git, and redacted from any error message.

> rule34 uses plural tags (`1girls`, `2girls`) and is mostly explicit — rating
> defaults auto-adjust per site.

## GPU acceleration

The ONNX tagger/encoder is the per-query bottleneck. See `requirements-gpu.txt`:

```bash
pip uninstall -y onnxruntime && pip install onnxruntime-gpu   # NVIDIA/CUDA
# Windows, any GPU: pip install onnxruntime-directml
```

Then (NVIDIA) launch with `ONNX_MODE=cuda`. The app prints the active ONNX providers
on startup so you can confirm the GPU is in use.

## Build your own index (e621 / safebooru)

- **Route A (fast)** — reuse deepghs precomputed embeddings via their
  [`embeddings_tools`](https://github.com/deepghs/embeddings_tools) (collect -> repack
  -> train), then point `INDEXES` in `app.py` at your new HF repo.
- **Route B (slow)** — embed a folder of images from scratch:
  `python build_index.py --images ./imgs --out indices/e621 --big`

## Files

| File | Purpose |
|------|---------|
| `app.py` | Gradio app: tabs, search handlers, settings, UI. |
| `resolver.py` | Per-site booru API: id->image, metadata, tag search, throttling, redaction. |
| `pools.py` | Generic cheesechaser mirror pool. |
| `build_index.py` | Build a FAISS index for a new site. |
| `test_pipeline.py` | Offline tests (FAISS pipeline + resolver parsing). |
| `requirements*.txt` | Pinned CPU deps / GPU instructions. |

## Development

```bash
python test_pipeline.py      # 17 offline tests, no network
```

CI runs the test suite on every push/PR (Python 3.11 & 3.12) via
`.github/workflows/ci.yml`.

## Credits

Indices, image mirrors, and the WD14/SigLIP models are by the
[deepghs](https://huggingface.co/deepghs) community; the original single-site approach
is by [SmilingWolf](https://huggingface.co/SmilingWolf). This project is the glue that
generalizes them into a self-hostable multi-booru app.

## Disclaimer

Several supported sites host adult content. Keep deployments access-controlled, follow
each site's API terms (rate limits; e621 requires a descriptive User-Agent), and use
responsibly.

## License

MIT — see [LICENSE](LICENSE).
