"""
ID -> live post URL resolvers for booru sites.

This is the "SmilingWolf style" retrieval path: given a numeric post id, hit the
site's public API, return the full-size image URL, and optionally filter by
rating. Use this when you want links back to the *original* post.

(The alternative path, used by deepghs/search_image_by_image, downloads a webp
copy from deepghs HuggingFace mirrors via `cheesechaser` instead of touching the
live site. See app.py RETRIEVAL_MODE.)

Every resolver takes (post_id, accepted_ratings, session) and returns a URL str
or None. `accepted_ratings` is a set drawn from the normalized vocabulary:
    {"general", "sensitive", "questionable", "explicit"}
Sites using the older 3-level scheme (safe/questionable/explicit) are mapped:
    safe -> general.

Parsing (`_extract_*`) is split from the HTTP call so it can be unit-tested
against captured JSON with no network.
"""
from __future__ import annotations

from typing import Optional, Set, Any
import requests

# A descriptive UA is *required* by e621 and good manners everywhere.
USER_AGENT = "booru_image_similarity/1.0 (self-hosted; contact: you@example.com)"

RATING_NORMALIZE = {
    "g": "general", "general": "general", "s": "sensitive", "sensitive": "sensitive",
    "q": "questionable", "questionable": "questionable", "e": "explicit", "explicit": "explicit",
    # legacy 3-level scheme used by rule34 / safebooru / old gelbooru
    "safe": "general",
}


def _norm_rating(raw: Optional[str]) -> Optional[str]:
    if raw is None:
        return None
    return RATING_NORMALIZE.get(str(raw).strip().lower())


def _passes(rating: Optional[str], accepted: Set[str]) -> bool:
    # If a site gives no rating, keep it only when the caller allows everything.
    if rating is None:
        return len(accepted) == 4
    return rating in accepted


# --------------------------------------------------------------------------- #
# Pure parsers: take decoded JSON body, return (url, normalized_rating).
# --------------------------------------------------------------------------- #
def _extract_gelbooru_style(body: Any):
    """gelbooru.com / rule34.xxx / safebooru.org dapi json.

    rule34/safebooru return a bare list; gelbooru returns {"post": [...]} (and
    {"@attributes":..., "post":[]} when empty). safebooru sometimes omits
    file_url and it must be built from directory + image.
    """
    if isinstance(body, dict):
        posts = body.get("post", [])
    else:
        posts = body
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
    rating = _norm_rating(post.get("rating"))
    url = (post.get("file") or {}).get("url")
    return url, rating


def _extract_danbooru(body: Any):
    """danbooru.donmai.us /posts/<id>.json -> flat object."""
    if not isinstance(body, dict) or not body:
        return None, None
    rating = _norm_rating(body.get("rating"))
    url = body.get("large_file_url") or body.get("file_url")
    return url, rating


# --------------------------------------------------------------------------- #
# Site registry: api url template + parser.
# --------------------------------------------------------------------------- #
SITES = {
    "rule34": {
        "api": "https://api.rule34.xxx/index.php?page=dapi&s=post&q=index&json=1&id={id}",
        "parser": _extract_gelbooru_style,
    },
    "gelbooru": {
        "api": "https://gelbooru.com/index.php?page=dapi&s=post&q=index&json=1&id={id}",
        "parser": _extract_gelbooru_style,
    },
    "safebooru": {
        "api": "https://safebooru.org/index.php?page=dapi&s=post&q=index&json=1&id={id}",
        "parser": _extract_gelbooru_style,
    },
    "e621": {
        "api": "https://e621.net/posts/{id}.json",
        "parser": _extract_e621,
    },
    "danbooru": {
        "api": "https://danbooru.donmai.us/posts/{id}.json",
        "parser": _extract_danbooru,
    },
}


def resolve(site: str, post_id: int, accepted_ratings: Set[str],
            session: Optional[requests.Session] = None,
            timeout: float = 10.0) -> Optional[str]:
    """Resolve one post id to a usable image URL, or None if blocked/missing."""
    cfg = SITES.get(site)
    if cfg is None:
        raise ValueError(f"unknown site {site!r}; known: {sorted(SITES)}")
    sess = session or requests.Session()
    url = cfg["api"].format(id=post_id)
    resp = sess.get(url, headers={"User-Agent": USER_AGENT}, timeout=timeout)
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
