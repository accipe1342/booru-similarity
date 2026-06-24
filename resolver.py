"""
Booru API helpers for the "live" path: resolve post ids -> image URLs, fetch a
post's metadata (url + rating + tags), and run native tag searches.

Politeness built in (so live mode won't get your IP throttled/banned):
  * per-site minimum interval between requests (throttle)
  * retry with exponential backoff on 429 / 503, honoring Retry-After

Ratings normalize to {"general","sensitive","questionable","explicit"};
legacy "safe" -> "general". Parsing is split from HTTP so it stays unit-testable.
"""
from __future__ import annotations

import re
import threading
import time
import urllib.parse
from typing import Optional, Set, Any, List, Dict

import requests

# e621 REQUIRES a descriptive UA with real contact info -- change this.
USER_AGENT = "booru_image_similarity/1.0 (self-hosted; contact: you@example.com)"

RATING_NORMALIZE = {
    "g": "general", "general": "general", "s": "sensitive", "sensitive": "sensitive",
    "q": "questionable", "questionable": "questionable", "e": "explicit", "explicit": "explicit",
    "safe": "general",
}

# Native tag-search caps. danbooru limits anonymous searches to 2 tags.
TAG_SEARCH_LIMIT = {"danbooru": 2, "rule34": 6, "gelbooru": 6, "safebooru": 6, "e621": 6,
                    "yandere": 6, "konachan": 6}

# Optional per-site auth appended to API URLs (gelbooru now needs api_key+user_id).
_CREDENTIALS: dict[str, str] = {}


def set_credentials(site: str, api_key: str = "", user_id: str = "") -> None:
    """Store gelbooru-style credentials; appended as &api_key=..&user_id=.. ."""
    api_key, user_id = (api_key or "").strip(), (user_id or "").strip()
    if api_key and user_id:
        _CREDENTIALS[site] = f"&api_key={api_key}&user_id={user_id}"
    else:
        _CREDENTIALS.pop(site, None)


def _cred(site: str) -> str:
    """Return the credential query-string suffix (e.g. '&api_key=..&user_id=..') for a site, or '' if none configured."""
    return _CREDENTIALS.get(site, "")


_REDACT_RE = re.compile(
    r"((?:api_key|user_id|login|token|password|api-key)=)[^&\s\"']+", re.I)


def redact(text) -> str:
    """Strip secrets from any string before it's shown/logged."""
    return _REDACT_RE.sub(r"\1***", str(text))


def _norm_rating(raw: Optional[str]) -> Optional[str]:
    """Normalize a raw site rating string to the canonical vocabulary (general/sensitive/questionable/explicit), or None if unrecognized."""
    return None if raw is None else RATING_NORMALIZE.get(str(raw).strip().lower())


def _passes(rating: Optional[str], accepted: Set[str]) -> bool:
    """Return True if a normalized rating is allowed by the accepted set. An unknown (None) rating passes only when all four ratings are accepted."""
    if rating is None:
        return len(accepted) == 4
    return rating in accepted


# --------------------------------------------------------------------------- #
# Per-POST extractors (operate on one post dict). kind = gelbooru|e621|danbooru
# --------------------------------------------------------------------------- #
def _purl_gelbooru(post):
    """Extract (image_url, rating) from one Gelbooru-style post, building the safebooru file_url from directory+image when file_url is absent."""
    url = post.get("file_url")
    if not url and post.get("directory") is not None and post.get("image"):
        url = f"https://safebooru.org/images/{post['directory']}/{post['image']}"
    return url, _norm_rating(post.get("rating"))


def _purl_e621(post):
    """Extract (image_url, rating) from one e621 post (file.url + rating)."""
    return (post.get("file") or {}).get("url"), _norm_rating(post.get("rating"))


def _purl_danbooru(post):
    """Extract (image_url, rating) from one danbooru post, preferring large_file_url over file_url."""
    return (post.get("large_file_url") or post.get("file_url")), _norm_rating(post.get("rating"))


# Moebooru (yande.re, konachan): rating "s" = SAFE (not sensitive).
_MOEBOORU_RATING = {"s": "general", "safe": "general", "q": "questionable",
                    "questionable": "questionable", "e": "explicit", "explicit": "explicit"}


def _purl_moebooru(post):
    """Extract (image_url, rating) from one Moebooru post. Note: Moebooru 's' means SAFE (-> general), unlike Gelbooru where 's' is sensitive."""
    r = str(post.get("rating", "")).strip().lower()
    return post.get("file_url"), _MOEBOORU_RATING.get(r)


def _ptags_moebooru(post):
    """Return the tag set from a Moebooru post (space-separated 'tags' string)."""
    return set((post.get("tags") or "").split())


def _ptags_gelbooru(post) -> Set[str]:
    """Return the tag set from a Gelbooru-style post (space-separated 'tags' string)."""
    return set((post.get("tags") or "").split())


def _ptags_e621(post) -> Set[str]:
    """Return the flattened tag set from an e621 post (union of all tag-category lists)."""
    t = post.get("tags") or {}
    return set(x for v in t.values() for x in (v or []))


def _ptags_danbooru(post) -> Set[str]:
    """Return the tag set from a danbooru post ('tag_string')."""
    return set((post.get("tag_string") or "").split())


def _pid(post):
    """Return a post's numeric id."""
    return post.get("id")


VIDEO_EXTS = {"webm", "mp4", "mov", "m4v", "avi", "mkv"}
GIF_EXTS = {"gif"}


def _ext_from_url(url) -> str:
    """Return the lowercase file extension from a URL path, or '' when there is none."""
    if not url:
        return ""
    path = urllib.parse.urlparse(url).path
    return path.rsplit(".", 1)[-1].lower() if "." in path else ""


def media_type(ext) -> str:
    """Classify a file extension as 'video', 'gif', or 'image'."""
    ext = (ext or "").lower().lstrip(".")
    if ext in VIDEO_EXTS:
        return "video"
    if ext in GIF_EXTS:
        return "gif"
    return "image"


def _post_meta(kind: str, post) -> Dict:
    """Full per-post info: url, rating, tags, ext, type (image/gif/video), preview."""
    url, rating = _PURL[kind](post)
    tags = _PTAGS[kind](post)
    if kind == "e621":
        ext = (post.get("file") or {}).get("ext") or _ext_from_url(url)
        preview = (post.get("sample") or {}).get("url") or (post.get("preview") or {}).get("url")
    elif kind == "danbooru":
        ext = post.get("file_ext") or _ext_from_url(url)
        preview = post.get("large_file_url") or post.get("preview_file_url")
    else:  # gelbooru-style + moebooru
        ext = _ext_from_url(url)
        preview = post.get("sample_url") or post.get("preview_url")
    return {"url": url, "rating": rating, "tags": tags,
            "ext": (ext or "").lower(), "type": media_type(ext), "preview": preview}


_PURL = {"gelbooru": _purl_gelbooru, "e621": _purl_e621, "danbooru": _purl_danbooru,
         "moebooru": _purl_moebooru}
_PTAGS = {"gelbooru": _ptags_gelbooru, "e621": _ptags_e621, "danbooru": _ptags_danbooru,
          "moebooru": _ptags_moebooru}


# --------------------------------------------------------------------------- #
# Body-level helpers: pull the post(s) out of a full API response by kind.
# --------------------------------------------------------------------------- #
def _posts_list(kind: str, body: Any) -> List[dict]:
    """Return a list of post dicts, tolerating single-object responses and junk.

    Boorus sometimes return one match as a bare object instead of a 1-element
    array (XML->JSON quirk), or wrap posts differently. Normalize all of it and
    drop anything that isn't a dict so callers never crash on a stray string.
    """
    if kind == "gelbooru":
        if isinstance(body, dict):
            p = body.get("post")
            if p is None:  # maybe the body itself is a single post
                p = body if ("id" in body or "file_url" in body) else []
        else:
            p = body
    elif kind == "e621":
        p = body.get("posts") if isinstance(body, dict) else body
    elif kind in ("danbooru", "moebooru"):
        p = body
    else:
        return []
    if isinstance(p, dict):
        p = [p]
    if not isinstance(p, list):
        return []
    return [x for x in p if isinstance(x, dict)]


def _single_post(kind: str, body: Any):
    """Return the first/only post dict from an API response, handling e621's singular 'post' key (vs the plural 'posts' used by searches)."""
    # e621's single-id endpoint uses singular "post"; search uses plural "posts".
    if kind == "e621":
        post = body.get("post") if isinstance(body, dict) else None
        return post if isinstance(post, dict) else None
    posts = _posts_list(kind, body)
    return posts[0] if posts else None


# Back-compat body-level extractors (used by resolve() and the unit tests).
def _extract_gelbooru_style(body: Any):
    """Body-level (image_url, rating) extractor for a Gelbooru-style single-id response."""
    p = _single_post("gelbooru", body)
    return _purl_gelbooru(p) if p else (None, None)


def _extract_e621(body: Any):
    """Body-level (image_url, rating) extractor for an e621 single-id response."""
    p = _single_post("e621", body)
    return _purl_e621(p) if p else (None, None)


def _extract_danbooru(body: Any):
    """Body-level (image_url, rating) extractor for a danbooru single-id response."""
    p = _single_post("danbooru", body)
    return _purl_danbooru(p) if p else (None, None)


def _extract_moebooru(body: Any):
    """Body-level (image_url, rating) extractor for a Moebooru single-id response."""
    p = _single_post("moebooru", body)
    return _purl_moebooru(p) if p else (None, None)


# --------------------------------------------------------------------------- #
# Site registry. api = single-id lookup; search = native tag search.
# --------------------------------------------------------------------------- #
SITES = {
    "rule34": {
        "kind": "gelbooru", "parser": _extract_gelbooru_style, "min_interval": 0.5,
        "api": "https://api.rule34.xxx/index.php?page=dapi&s=post&q=index&json=1&id={id}",
        "search": "https://api.rule34.xxx/index.php?page=dapi&s=post&q=index&json=1&limit={limit}&tags={tags}",
    },
    "gelbooru": {
        "kind": "gelbooru", "parser": _extract_gelbooru_style, "min_interval": 0.5,
        "api": "https://gelbooru.com/index.php?page=dapi&s=post&q=index&json=1&id={id}",
        "search": "https://gelbooru.com/index.php?page=dapi&s=post&q=index&json=1&limit={limit}&tags={tags}",
    },
    "safebooru": {
        "kind": "gelbooru", "parser": _extract_gelbooru_style, "min_interval": 0.5,
        "api": "https://safebooru.org/index.php?page=dapi&s=post&q=index&json=1&id={id}",
        "search": "https://safebooru.org/index.php?page=dapi&s=post&q=index&json=1&limit={limit}&tags={tags}",
    },
    "e621": {
        "kind": "e621", "parser": _extract_e621, "min_interval": 1.0,
        "api": "https://e621.net/posts/{id}.json",
        "search": "https://e621.net/posts.json?limit={limit}&tags={tags}",
    },
    "danbooru": {
        "kind": "danbooru", "parser": _extract_danbooru, "min_interval": 0.15,
        "api": "https://danbooru.donmai.us/posts/{id}.json",
        "search": "https://danbooru.donmai.us/posts.json?limit={limit}&tags={tags}",
    },
    "yandere": {
        "kind": "moebooru", "parser": _extract_moebooru, "min_interval": 1.0,
        "api": "https://yande.re/post.json?tags=id:{id}",
        "search": "https://yande.re/post.json?limit={limit}&tags={tags}",
    },
    "konachan": {
        "kind": "moebooru", "parser": _extract_moebooru, "min_interval": 1.0,
        "api": "https://konachan.com/post.json?tags=id:{id}",
        "search": "https://konachan.com/post.json?limit={limit}&tags={tags}",
    },
}

# --------------------------------------------------------------------------- #
# Per-site throttle (thread-safe).
# --------------------------------------------------------------------------- #
_last_call: dict[str, float] = {}
_locks: dict[str, threading.Lock] = {}
_locks_guard = threading.Lock()


def _site_lock(site: str) -> threading.Lock:
    """Return (creating on first use) the per-site lock that serializes throttling for that site."""
    with _locks_guard:
        return _locks.setdefault(site, threading.Lock())


def _throttle(site: str, min_interval: float):
    """Block until at least min_interval seconds have elapsed since the last request to this site (thread-safe, per site)."""
    with _site_lock(site):
        wait = min_interval - (time.monotonic() - _last_call.get(site, 0.0))
        if wait > 0:
            time.sleep(wait)
        _last_call[site] = time.monotonic()


def _request_json(site: str, url: str, session: requests.Session,
                  timeout: float, max_retries: int):
    """Throttled GET with retry/backoff. Returns decoded JSON or None."""
    min_interval = SITES[site].get("min_interval", 0.5)
    backoff = 1.0
    for attempt in range(max_retries + 1):
        _throttle(site, min_interval)
        try:
            resp = session.get(url, headers={"User-Agent": USER_AGENT}, timeout=timeout)
        except requests.RequestException:
            if attempt >= max_retries:
                return None
            time.sleep(backoff); backoff *= 2; continue
        if resp.status_code in (429, 503) and attempt < max_retries:
            ra = resp.headers.get("Retry-After")
            try:
                delay = float(ra) if ra else backoff
            except ValueError:
                delay = backoff
            time.sleep(delay); backoff *= 2; continue
        if resp.status_code != 200:
            return None
        try:
            return resp.json()
        except ValueError:
            return None
    return None


def build_tag_search_url(site: str, tags: List[str], limit: int) -> str:
    """Pure URL builder (unit-testable). Spaces->underscores, url-encoded, '+'-joined."""
    cfg = SITES[site]
    cleaned = [urllib.parse.quote(t.strip().replace(" ", "_"), safe="") for t in tags if t.strip()]
    return cfg["search"].format(limit=limit, tags="+".join(cleaned)) + _cred(site)


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def resolve(site: str, post_id: int, accepted_ratings: Set[str],
            session: Optional[requests.Session] = None,
            timeout: float = 10.0, max_retries: int = 3) -> Optional[str]:
    """Single id -> image URL (or None), filtered by rating."""
    cfg = SITES.get(site)
    if cfg is None:
        raise ValueError(f"unknown site {site!r}; known: {sorted(SITES)}")
    sess = session or requests.Session()
    body = _request_json(site, cfg["api"].format(id=post_id) + _cred(site), sess, timeout, max_retries)
    if body is None:
        return None
    url, rating = cfg["parser"](body)
    if url is None:
        return None
    return url if _passes(rating, accepted_ratings) else None


def fetch_meta(site: str, post_id: int,
               session: Optional[requests.Session] = None,
               timeout: float = 10.0, max_retries: int = 3) -> Optional[Dict]:
    """Single id -> {'url','rating','tags'} (no rating filter; caller decides)."""
    cfg = SITES.get(site)
    if cfg is None:
        raise ValueError(f"unknown site {site!r}; known: {sorted(SITES)}")
    sess = session or requests.Session()
    body = _request_json(site, cfg["api"].format(id=post_id) + _cred(site), sess, timeout, max_retries)
    if body is None:
        return None
    post = _single_post(cfg["kind"], body)
    if not post:
        return None
    return {"id": _pid(post), **_post_meta(cfg["kind"], post)}


def tag_search(site: str, tags: List[str], limit: int, accepted_ratings: Set[str],
               session: Optional[requests.Session] = None,
               timeout: float = 10.0, max_retries: int = 3) -> List[Dict]:
    """Native booru tag search -> list of {'id','url','rating','tags'} (rating-filtered)."""
    cfg = SITES.get(site)
    if cfg is None:
        raise ValueError(f"unknown site {site!r}; known: {sorted(SITES)}")
    if not tags:
        return []
    sess = session or requests.Session()
    url = build_tag_search_url(site, tags, limit)
    body = _request_json(site, url, sess, timeout, max_retries)
    if body is None:
        return []
    out = []
    for post in _posts_list(cfg["kind"], body):
        meta = _post_meta(cfg["kind"], post)
        if meta["url"] and _passes(meta["rating"], accepted_ratings):
            out.append({"id": _pid(post), **meta})
    return out


# --------------------------------------------------------------------------- #
# Original post-page URLs (the human-viewable post, not the raw image).
# --------------------------------------------------------------------------- #
POST_URL = {
    "rule34": "https://rule34.xxx/index.php?page=post&s=view&id={id}",
    "gelbooru": "https://gelbooru.com/index.php?page=post&s=view&id={id}",
    "safebooru": "https://safebooru.org/index.php?page=post&s=view&id={id}",
    "e621": "https://e621.net/posts/{id}",
    "danbooru": "https://danbooru.donmai.us/posts/{id}",
    "yandere": "https://yande.re/post/show/{id}",
    "konachan": "https://konachan.com/post/show/{id}",
    "zerochan": "https://www.zerochan.net/{id}",
    "anime_pictures": "https://anime-pictures.net/posts/{id}",
}


def post_url(site: str, post_id) -> Optional[str]:
    """Return the human-viewable post-page URL for a site and post id, or None if the site is unknown."""
    tmpl = POST_URL.get(site)
    return tmpl.format(id=post_id) if tmpl else None
