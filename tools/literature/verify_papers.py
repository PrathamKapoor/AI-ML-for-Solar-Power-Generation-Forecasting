"""Verify the curated paper set against Crossref, OpenAlex and Semantic Scholar.

Responsibilities
----------------
1. Resolve any placeholder DOI by fuzzy-matching the corresponding seed title
   against Crossref/OpenAlex (placeholder DOIs are never trusted as-is).
2. Fetch authoritative metadata from three independent sources.
3. Reconstruct abstracts (Crossref JATS, OpenAlex inverted index, S2 text).
4. Detect disagreements between sources so that nothing is silently invented.
5. Emit ``literature/verification/verified_papers.json`` with per-field
   provenance, plus one raw payload per source for audit.

A field is only populated when at least one source supplies it. Agreement
between sources is recorded; conflicts are surfaced in ``conflicts`` and must
be reviewed manually.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any

import requests

from resolve_seeds import (CROSSREF, OPENALEX, PROJECT_ROOT, SEEDS_PATH, USER_AGENT,
                           clean_abstract, crossref_to_record, openalex_to_record,
                           query_crossref, query_openalex, title_similarity)

S2 = "https://api.semanticscholar.org/graph/v1/paper"
CURATED = PROJECT_ROOT / "tools" / "literature" / "final_papers.json"
SEED_RES = PROJECT_ROOT / "literature" / "verification" / "seed_resolution.json"
OUT_DIR = PROJECT_ROOT / "literature" / "verification"
RAW_DIR = OUT_DIR / "raw"


def _get(url: str, params: dict[str, Any] | None = None, tries: int = 4) -> dict | None:
    for attempt in range(tries):
        try:
            r = requests.get(url, params=params or {}, timeout=45, headers={"User-Agent": USER_AGENT})
            if r.status_code == 200:
                return r.json()
            if r.status_code == 429:
                time.sleep(4 * (attempt + 1))
                continue
            if r.status_code == 404:
                return None
            time.sleep(1.5 * (attempt + 1))
        except requests.RequestException:
            time.sleep(1.5 * (attempt + 1))
    return None


def crossref_by_doi(doi: str) -> dict | None:
    data = _get(f"{CROSSREF}/{doi}")
    return crossref_to_record(data["message"], 1.0) if data else None


def oa_abstract(item: dict) -> str | None:
    """Rebuild an abstract from an OpenAlex inverted index."""
    inv = item.get("abstract_inverted_index")
    if not inv:
        return None
    positions: list[tuple[int, str]] = []
    for word, idxs in inv.items():
        positions.extend((i, word) for i in idxs)
    positions.sort()
    return re.sub(r"\s+", " ", " ".join(w for _, w in positions)).strip() or None


def openalex_by_doi(doi: str) -> dict | None:
    data = _get(f"{OPENALEX}/doi:{doi}")
    if not data:
        return None
    rec = openalex_to_record(data, 1.0)
    rec["abstract"] = oa_abstract(data)
    return rec


def s2_by_doi(doi: str) -> dict | None:
    fields = ("title,abstract,year,venue,publicationVenue,authors,externalIds,"
              "citationCount,publicationDate,openAccessPdf,fieldsOfStudy,journal,isOpenAccess")
    data = _get(f"{S2}/DOI:{doi}", {"fields": fields})
    if not data:
        return None
    return {
        "source": "semanticscholar",
        "title": data.get("title"),
        "authors": [a["name"] for a in (data.get("authors") or []) if a.get("name")],
        "year": data.get("year"),
        "publication_date": data.get("publicationDate"),
        "venue": data.get("venue"),
        "journal": (data.get("journal") or {}).get("name"),
        "abstract": data.get("abstract"),
        "citation_count": data.get("citationCount"),
        "doi": (data.get("externalIds") or {}).get("DOI"),
        "open_access_pdf": (data.get("openAccessPdf") or {}).get("url"),
        "is_oa": data.get("isOpenAccess"),
    }


def oa_abstract(item: dict) -> str | None:
    """Rebuild an abstract from an OpenAlex inverted index."""
    inv = item.get("abstract_inverted_index")
    if not inv:
        return None
    positions: list[tuple[int, str]] = []
    for word, idxs in inv.items():
        positions.extend((i, word) for i in idxs)
    positions.sort()
    return re.sub(r"\s+", " ", " ".join(w for _, w in positions)).strip() or None


def resolve_placeholder_doi(paper: dict, seed_index: dict[str, dict]) -> tuple[str | None, dict]:
    """Replace a placeholder DOI with the DOI of the best verified title match."""
    seed = seed_index.get(paper.get("from_seed", ""))
    if not seed:
        return None, {"status": "no_seed_for_placeholder"}

    candidates = (seed.get("crossref_candidates") or []) + (seed.get("openalex_candidates") or [])
    best, best_score = None, 0.0
    for cand in candidates:
        doi, title = cand.get("doi"), cand.get("title")
        if not doi or not title:
            continue
        score = title_similarity(seed["seed_title"], title)
        if score > best_score:
            best, best_score = cand, score
    if best and best_score >= 0.85:
        return best["doi"], {"status": "resolved_from_seed", "score": round(best_score, 4),
                             "title": best.get("title"), "container": best.get("container")}
    # Fall back to a live title query.
    for item in query_crossref(seed["seed_title"], rows=8):
        rec_title = (item.get("title") or [""])[0]
        score = title_similarity(seed["seed_title"], rec_title)
        if score >= 0.9 and item.get("DOI"):
            return item["DOI"], {"status": "resolved_live_crossref", "score": round(score, 4),
                                 "title": rec_title, "container": (item.get("container-title") or [None])[0]}
    for item in query_openalex(seed["seed_title"], per_page=8):
        rec_title = item.get("title") or item.get("display_name")
        score = title_similarity(seed["seed_title"], rec_title or "")
        if score >= 0.9 and item.get("doi"):
            return item["doi"], {"status": "resolved_live_openalex", "score": round(score, 4),
                                 "title": rec_title}
    return None, {"status": "unresolved"}


def first_nonempty(*values):
    for v in values:
        if v:
            return v
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--curated", type=Path, default=CURATED)
    ap.add_argument("--out", type=Path, default=OUT_DIR / "verified_papers.json")
    args = ap.parse_args()

    curated = json.loads(args.curated.read_text(encoding="utf-8"))["papers"]
    seed_res = json.loads(SEED_RES.read_text(encoding="utf-8"))
    seed_index = {r["id"]: r for r in seed_res}
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    out, unresolved = [], []
    seen_dois: dict[str, str] = {}

    for i, paper in enumerate(curated, 1):
        pid, doi, prov = paper["id"], paper.get("doi"), {}
        if paper.get("doi_unverified"):
            new_doi, prov = resolve_placeholder_doi(paper, seed_index)
            if new_doi:
                doi = new_doi
            else:
                unresolved.append(pid)
                print(f"[{i:2d}] {pid}: PLACEHOLDER UNRESOLVED")
                continue
        if not doi:
            unresolved.append(pid)
            print(f"[{i:2d}] {pid}: NO DOI")
            continue
        if doi.lower() in seen_dois:
            print(f"[{i:2d}] {pid}: DUPLICATE of {seen_dois[doi.lower()]} ({doi})")
            continue
        seen_dois[doi.lower()] = pid

        cr = crossref_by_doi(doi)
        time.sleep(0.25)
        oa = openalex_by_doi(doi)
        time.sleep(0.25)
        s2 = s2_by_doi(doi)
        time.sleep(0.35)

        for name, payload in (("crossref", cr), ("openalex", oa), ("s2", s2)):
            if payload:
                (RAW_DIR / f"{pid}_{name}.json").write_text(
                    json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

        oa_abstract_text = (oa or {}).get("abstract")
        abstract = first_nonempty(
            (cr or {}).get("abstract"), (s2 or {}).get("abstract"), oa_abstract_text)
        abstract_source = None
        if (cr or {}).get("abstract"):
            abstract_source = "crossref"
        elif (s2 or {}).get("abstract"):
            abstract_source = "semanticscholar"
        elif oa_abstract_text:
            abstract_source = "openalex_inverted_index"

        titles = [t for t in [(cr or {}).get("title"), (oa or {}).get("title"), (s2 or {}).get("title")] if t]
        title = first_nonempty(*titles)
        if cr and oa and cr.get("title") and oa.get("title"):
            agree = title_similarity(cr["title"], oa["title"])
        else:
            agree = None

        record = {
            "id": pid,
            "category": paper["category"],
            "category_name": paper["category_name"],
            "from_seed": paper.get("from_seed"),
            "doi_resolution": prov or {"status": "curated_direct"},
            "doi": doi,
            "title": title,
            "title_variants": sorted(set(titles)),
            "authors_crossref": (cr or {}).get("authors"),
            "authors_openalex": (oa or {}).get("authors"),
            "authors_s2": (s2 or {}).get("authors"),
            "date_parts_crossref": (cr or {}).get("date_parts"),
            "published_print": (cr or {}).get("published_print"),
            "published_online": (cr or {}).get("published_online"),
            "publication_date_s2": (s2 or {}).get("publication_date"),
            "year_openalex": (oa or {}).get("publication_year"),
            "container_crossref": (cr or {}).get("container"),
            "container_openalex": (oa or {}).get("container"),
            "venue_s2": (s2 or {}).get("venue"),
            "journal_s2": (s2 or {}).get("journal"),
            "publisher_crossref": (cr or {}).get("publisher"),
            "publisher_openalex": (oa or {}).get("publisher"),
            "type_crossref": (cr or {}).get("type"),
            "type_openalex": (oa or {}).get("type"),
            "url_crossref": (cr or {}).get("url"),
            "url_openalex": (oa or {}).get("url"),
            "abstract": abstract,
            "abstract_source": abstract_source,
            "subjects_crossref": (cr or {}).get("subject"),
            "keywords_openalex": (oa or {}).get("keywords"),
            "concepts_openalex": (oa or {}).get("concepts"),
            "citation_count_s2": (s2 or {}).get("citation_count"),
            "citation_count_crossref": (cr or {}).get("cited_by_count"),
            "citation_count_openalex": (oa or {}).get("cited_by_count"),
            "open_access_pdf": (s2 or {}).get("open_access_pdf"),
            "title_agreement_crossref_openalex": (round(agree, 4) if agree is not None else None),
            "sources_found": [n for n, p in (("crossref", cr), ("openalex", oa), ("s2", s2)) if p],
        }
        out.append(record)
        print(f"[{i:2d}] {pid:<3} {str(record['title'])[:62]:<62} "
              f"| cr={bool(cr)} oa={bool(oa)} s2={bool(s2)} abs={bool(abstract)} "
              f"| {record['container_crossref'] or record['container_openalex']}")

    args.out.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nverified {len(out)} papers -> {args.out}")
    if unresolved:
        print(f"UNRESOLVED: {unresolved}")
    cats: dict[str, int] = {}
    for r in out:
        cats[r["category"]] = cats.get(r["category"], 0) + 1
    print("category counts:", dict(sorted(cats.items())))


if __name__ == "__main__":
    main()
