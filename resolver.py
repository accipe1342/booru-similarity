"""
ID -> live post URL resolvers for booru sites (the "live" retrieval path).

Given a numeric post id, hit the site's public API, return the full-size image
URL, and optionally filter by rating. Use this when you want links back to the
*original* post. (Mirror mode in app.py instead pulls webp copies from deepghs
HF mirrors and never touches the live site.)

Built-in politeness so live mode won't get your IP throttled/banned:
  * per-site minimum interval between requests (token-bucket-ish throttle)
  * automatic retry with exponential backoff on 429 / 503
  * honors the server's Retry-After header
Tune the limits in SITES[...]["min_interval"] if a site allows more/less.

Parsing (`_extract_*`) is split from HTTP so it stays unit-testable with no
network. Ratings normalize to {"general","sensitive","questionable","explicit"};
the legacy "safe" maps to "general".
"""
from __future__ import annotations

import threading
import time
from typing import Optional, Set, Any

import requests

# e621 REQUIRES a descriptive UA with real contact info -- change this.
USER_AGENT = "booru_image_similarity/1.0 (self-hosted; contact: you@example.com)"

RATING_NORMALIZE = {
    "g": "general", "general": "general", "s": "sensitive", "sensitive": "sensitive",
    "q": "questionable", "questionable": "questionable", "e": "explicit", "explicit": "explicit",
    "safe": "general",  # legacy 3-level scheme (rule34 / safebooru / old gelbooru)
}


def _norm_rating(raw: Optional[str]) -> Optional[str]:
    if raw is None:
        return None
    return RATING_NORMALIZE.get(str(raw).strip().lower())


def _passes(rating: Optional[str], accepted: Set[str]) -> bool:
    if rating is None:
        return len(accepted) == 4  # unknown rating: keep only if everything allowed
    return rating in accepted


# --------------------------------------------------------------------------- #
# Pure parsers: decoded JSON body -> (url, normalized_rating).
# --------------------------------------------------------------------------- #
def _extract_gelbooru_style(body: Any):
    """gelbooru.com / rule34.xxx / safebooru.org dapi json."""
    posts = body.get("post", []) if isinstance(body, dict) else body
    if not posts:
        return None, None
    post = posts[0]
    rating = _norm_rating(post.get("rating"))
    url = post.get("file_url")
    if not url and post.get("directory") is not None and post.get("image"):
        url = f"https://safebooru.org/images/{post['directory']}/{post['image']}"
    return url, rating


def _extract_e621(body: Any):
    """e621.net /posts/<id>.json -> {"post": {"file": {"url": ...}, "rating": "s"}}"""
    post = body.get("post") if isinstance(body, dict) else None
    if not post:
        return None, None
    return (post.get("file") or {}).get("url"), _norm_rating(post.get("rating"))


def _extract_danbooru(body: Any):
    """danbooru.donmai.us /posts/<id>.json -> flat object."""
    if not isinstance(body, dict) or not body:
        return None, None
    return (body.get("large_file_url") or body.get("file_url")), _norm_rating(body.get("rating"))


# --------------------------------------------------------------------------- #
# Site registry. min_interval = seconds to wait between requests to that site.
# --------------------------------------------------------------------------- #
SITES = {
    "rule34": {
        "api": "https://api.rule34.xxx/index.php?page=dapi&s=post&q=index&json=1&id={id}",
        "parser": _extract_gelbooru_style, "min_interval": 0.5,
    },
    "gelbooru": {
        "api": "https://gelbooru.com/index.php?page=dapi&s=post&q=index&json=1&id={id}",
        "parser": _extract_gelbooru_style, "min_interval": 0.5,
    },
    "safebooru": {
        "api": "https://safebooru.org/index.php?page=dapi&s=post&q=index&json=1&id={id}",
        "parser": _extract_gelbooru_style, "min_interval": 0.5,
    },
    "e621": {  # strictest: keep ~1 req/sec and a real UA, or you WILL get blocked
        "api": "https://e621.net/posts/{id}.json",
        "parser": _extract_e621, "min_interval": 1.0,
    },
    "danbooru": {
        "api": "https://danbooru.donmai.us/posts/{id}.json",
        "parser": _extract_danbooru, "min_interval": 0.15,
    },
}

# --------------------------------------------------------------------------- #
# Per-site throttle (thread-safe). Spaces out calls to the same host.
# --------------------------------------------------------------------------- #
_last_call: dict[str, float] = {}
_locks: dict[str, threading.Lock] = {}
_locks_guard = threading.Lock()


def _site_lock(site: str) -> threading.Lock:
    with _locks_guard:
        if site not in _locks:
            _locks[site] = threading.Lock()
        return _locks[site]


def _throttle(site: str, min_interval: float):
    with _site_lock(site):
        now = time.monotonic()
        wait = min_interval - (now - _last_call.get(site, 0.0))
        if wait > 0:
            time.sleep(wait)
        _last_call[site] = time.monotonic()


def resolve(site: str, post_id: int, accepted_ratings: Set[str],
            session: Optional[requests.Session] = None,
            timeout: float = 10.0, max_retries: int = 3) -> Optional[str]:
    """Resolve one post id to an image URL, or None if blocked/missing.

    Polite by default: throttles per site and retries 429/503 with backoff.
    """
    cfg = SITES.get(site)
    if cfg is None:
        raise ValueError(f"unknown site {site!r}; known: {sorted(SITES)}")
    sess = session or requests.Session()
    url = cfg["api"].format(id=post_id)
    min_interval = cfg.get("min_interval", 0.5)

    backoff = 1.0
    for attempt in range(max_retries + 1):
        _throttle(site, min_interval)
        try:
            resp = sess.get(url, headers={"User-Agent": USER_AGENT}, timeout=timeout)
        except requests.RequestException:
            if attempt >= max_retries:
                return None
            time.sleep(backoff); backoff *= 2
            continue

        if resp.status_code in (429, 503) and attempt < max_retries:
            # respect Retry-After if present, else exponential backoff
            retry_after = resp.headers.get("Retry-After")
            try:
                delay = float(retry_after) if retry_after else backoff
            except ValueError:
                delay = backoff
            time.sleep(delay); backoff *= 2
            continue

        if resp.status_code != 200:
            return None
        try:
            body = resp.json()
        except ValueError:
            return None
        img_url, rating = cfg["parser"](body)
        if img_url is None:
            return None
        return img_url if _passes(rating, accepted_ratings) else None

    return None
