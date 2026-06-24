# Changelog

All notable changes to this project are documented here.
This project adheres to [Semantic Versioning](https://semver.org/).

## [1.0.0] - 2026-06-24

First tagged release.

### Added
- Visual similarity search across 8 prebuilt deepghs indices (rule34, gelbooru,
  danbooru, yandere, konachan, zerochan, anime_pictures, and a ~78M all-in-one).
- Text and blended (image + text) search via the SigLIP model that shares the WD14
  embedding space — reusing the existing indices.
- Native booru tag search with per-site tag caps and tag-overlap rerank.
- `mirror` (HuggingFace mirrors) and `live` (booru API) retrieval modes.
- Moebooru (yande.re, konachan) live + tag support, including the `s`=safe rating quirk.
- Detected-tag chips, a tag-confidence dropdown, and clickable result thumbnails that
  open the original post; video preview thumbnails + media-type filtering.
- Per-site rating defaults, polite throttling with retry/backoff, and a Settings tab
  for the HF token, e621 contact, and rule34/gelbooru credentials (optionally saved).
- Credential redaction in all error output; auto-deletion of cached result images.
- GitHub Actions CI, GPU (ONNX) support, pinned requirements, and an index builder.

### Notes
- Indices, mirrors, and models are provided by deepghs and fetched at runtime.
- rule34 and gelbooru now require API credentials for live/tag search.
