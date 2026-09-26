"""Resolve literature seed titles against authoritative metadata sources.

Discovery seeds are *hints*, not citations. This module queries Crossref and
OpenAlex, scores candidate records by normalised title similarity, and stores
the best match together with the unmodified API payloads so that every
downstream field can be traced back to a source response.

Outputs
-------
literature/verification/seed_resolution.json   best match per seed + scores
literature/verification/raw/<id>_<source>.json  unmodified API payloads

No field is ever invented here: if a source omits a field it stays ``None``.
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
import time
import unicodedata
from pathlib import Path
from typing import Any

import requests

USER_AGENT = "solar-pv-forecasting-review/1.0 (mailto:prathamkapoor027@gmail.com)"
CROSSREF = "https://api.crossref.org/works"
OPENALEX = "https://api.openalex.org/works"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SEEDS_PATH = PROJECT_ROOT / "tools" / "literature" / "seed_candidates.json"
OUT_DIR = PROJECT_ROOT / "literature" / "verification"
RAW_DIR = OUT_DIR / "raw"

STOPWORDS = {
    "a", "an", "the", "of", "for", "and", "in", "on", "to", "with", "based",
    "using", "via", "approach", "model", "models", "method", "methods",
}


def normalise_title(text: str) -> str:
    """Lowercase, strip accents/punctuation, collapse whitespace."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def content_tokens(text: str) -> set[str]:
    return {t for t in normalise_title(text).split() if t not in STOPWORDS and len(t) > 2}


def title_similarity(a: str, b: str) -> float:
    """Blend sequence ratio with token-set Jaccard for robustness to word order."""
    na, nb = normalise_title(a), normalise_title(b)
    if not na or not nb:
        return 0.0
    seq = difflib.SequenceMatcher(None, na, nb).ratio()
    ta, tb = content_tokens(a), content_tokens(b)
    jac = len(ta & tb) / len(ta | tb) if (ta | tb) else 0.0
    return 0.5 * seq + 0.5 * jac


def _get(url: str, params: dict[str, Any], tries: int = 4) -> dict | None:
    for attempt in range(tries):
        try:
            resp = requests.get(url, params=params, timeout=45, headers={"User-Agent": USER_AGENT})
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code in (429, 500, 502, 503, 504):
                time.sleep(2 * (attempt + 1))
                continue
            return None
        except requests.RequestException:
            time.sleep(2 * (attempt + 1))
    return None


def query_crossref(title: str, rows: int = 8) -> list[dict]:
    data = _get(CROSSREF, {"query.bibliographic": title, "rows": rows,
                           "select": "DOI,title,author,issued,container-title,publisher,type,URL,abstract,subject,link,is-referenced-by-count"})
    if not data:
        return []
    return [it for it in data.get("message", {}).get("items", []) if it.get("title")]


def query_openalex(title: str, per_page: int = 8) -> list[dict]:
    data = _get(OPENALEX, {"filter": "title.search:" + normalise_title(title)[:180],
                           "per-page": per_page})
    if data:
        return data.get("results", [])
    data = _get(OPENALEX, {"search": title, "per-page": per_page})
    return data.get("results", []) if data else []


def clean_abstract(raw: str | None) -> str | None:
    """Crossref abstracts are JATS XML fragments; strip tags and entities."""
    if not raw:
        return None
    text = re.sub(r"<[^>]+>", " ", raw)
    text = (text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
                .replace("&quot;", '"').replace("&apos;", "'").replace("&nbsp;", " "))
    text = re.sub(r"\s+", " ", text).strip()
    return text or None


def crossref_to_record(item: dict, score: float) -> dict:
    titles = item.get("title") or []
    title = titles[0] if titles else None
    authors = []
    for a in item.get("author", []) or []:
        given, family = a.get("given"), a.get("family")
        if family:
            authors.append(f"{given} {family}".strip() if given else family)
        elif a.get("name"):
            authors.append(a["name"])
    issued = item.get("issued", {}).get("date-parts", [[]])[0]
    containers = item.get("container-title") or []
    return {
        "source": "crossref",
        "match_score": round(score, 4),
        "title": title,
        "authors": authors,
        "date_parts": issued,
        "published_print": (item.get("published-print", {}).get("date-parts", [[]])[0] or None),
        "published_online": (item.get("published-online", {}).get("date-parts", [[]])[0] or None),
        "container": containers[0] if containers else None,
        "publisher": item.get("publisher"),
        "type": item.get("type"),
        "doi": item.get("DOI"),
        "url": item.get("URL"),
        "abstract": clean_abstract(item.get("abstract")),
        "subject": item.get("subject") or [],
        "cited_by_count": item.get("is-referenced-by-count"),
        "is_referenced_by_count": item.get("is-referenced-by-count"),
    }


def openalex_to_record(item: dict, score: float) -> dict:
    loc = item.get("primary_location") or {}
    src = loc.get("source") or {}
    authors = [
        (a.get("author") or {}).get("display_name")
        for a in item.get("authorships", [])
        if (a.get("author") or {}).get("display_name")
    ]
    oa = item.get("open_access") or {}
    return {
        "source": "openalex",
        "match_score": round(score, 4),
        "title": item.get("title") or item.get("display_name"),
        "authors": authors,
        "date_parts": None,
        "publication_date": item.get("publication_date"),
        "publication_year": item.get("publication_year"),
        "container": src.get("display_name"),
        "publisher": src.get("host_organization_name"),
        "type": item.get("type"),
        "doi": item.get("doi"),
        "url": loc.get("landing_page_url") or item.get("doi"),
        "abstract": None,
        "concepts": [c.get("display_name") for c in (item.get("concepts") or [])[:8]],
        "keywords": [k.get("display_name") for k in (item.get("keywords") or [])[:8]],
        "cited_by_count": item.get("cited_by_count"),
        "is_oa": oa.get("is_oa"),
        "openalex_id": item.get("id"),
    }


def best_match(records: list[dict]) -> dict | None:
    return max(records, key=lambda r: r["match_score"]) if records else None


def resolve_one(seed: dict) -> dict:
    sid, title = seed["id"], seed["seed_title"]
    cr_items, oa_items = query_crossref(title), query_openalex(title)

    cr_records = [crossref_to_record(it, title_similarity(title, (it.get("title") or [""])[0]))
                  for it in cr_items]
    oa_records = [openalex_to_record(it, title_similarity(title, it.get("title") or it.get("display_name") or ""))
                  for it in oa_items]
    for rec_list in (cr_records, oa_records):
        rec_list.sort(key=lambda r: r["match_score"], reverse=True)

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if cr_items:
        (RAW_DIR / f"{sid}_crossref.json").write_text(json.dumps(cr_items, indent=2), encoding="utf-8")
    if oa_items:
        (RAW_DIR / f"{sid}_openalex.json").write_text(json.dumps(oa_items[:5], indent=2), encoding="utf-8")

    return {
        "id": sid,
        "seed_title": title,
        "category_seed": seed.get("category"),
        "crossref_best": best_match(cr_records),
        "crossref_candidates": cr_records[:3],
        "openalex_best": best_match(oa_records),
        "openalex_candidates": oa_records[:3],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=Path, default=SEEDS_PATH)
    ap.add_argument("--out", type=Path, default=OUT_DIR / "seed_resolution.json")
    args = ap.parse_args()

    seeds = json.loads(args.seeds.read_text(encoding="utf-8"))["seeds"]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    resolutions = []
    for i, seed in enumerate(seeds, 1):
        res = resolve_one(seed)
        resolutions.append(res)
        cr, oa = res["crossref_best"], res["openalex_best"]
        print(f"[{i:2d}/{len(seeds)}] {seed['id']:<34} "
              f"crossref={cr['match_score'] if cr else 0:.3f} "
              f"openalex={oa['match_score'] if oa else 0:.3f}  "
              f"{(cr or oa or {}).get('title', '')[:70]}")
        time.sleep(0.4)

    args.out.write_text(json.dumps(resolutions, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
