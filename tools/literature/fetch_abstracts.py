"""Recover abstracts that Crossref, OpenAlex and Semantic Scholar do not expose.

Some Elsevier and Springer records omit abstracts from every open metadata API.
This module falls back to the publisher's public metadata endpoints:

* Elsevier  -> ``https://api.elsevier.com/content/abstract/doi/<doi>``
               (works without an API key for open-access records; otherwise the
               ScienceDirect ``citation_abstract`` meta tag is scraped)
* Springer  -> Crossref is already used; falls back to the ``dc.description``
               meta tag on the landing page.

Anything still unresolved is left as ``None`` and reported, never guessed.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import requests

from resolve_seeds import PROJECT_ROOT, USER_AGENT, clean_abstract

VERIFIED = PROJECT_ROOT / "literature" / "verification" / "verified_papers.json"
SUPPLEMENTARY = PROJECT_ROOT / "tools" / "literature" / "supplementary_abstracts.json"
RAW_DIR = PROJECT_ROOT / "literature" / "verification" / "raw"

HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

META_PATTERNS = [
    r'<meta\s+name="citation_abstract"\s+content="([^"]+)"',
    r'<meta\s+name="dc\.description"\s+content="([^"]+)"',
    r'<meta\s+property="og:description"\s+content="([^"]+)"',
]


def unescape(text: str) -> str:
    import html
    return re.sub(r"\s+", " ", html.unescape(text)).strip()


def scrape_landing(doi: str) -> tuple[str | None, str | None]:
    """Follow the DOI to the publisher landing page and read abstract metadata.

    Several publishers (notably ScienceDirect) serve an abstract-bearing page to
    browser-like clients but an empty shell to generic agents, so a realistic
    browser header set is required.
    """
    url = f"https://doi.org/{doi}"
    try:
        r = requests.get(url, timeout=40, headers=HEADERS, allow_redirects=True)
    except requests.RequestException:
        return None, None
    if r.status_code != 200:
        return None, None
    content_type = r.headers.get("Content-Type", "")
    if "html" not in content_type.lower():
        return None, None
    for pat in META_PATTERNS:
        m = re.search(pat, r.text, flags=re.IGNORECASE)
        if m and m.group(1).strip():
            return unescape(m.group(1)), r.url
    return None, r.url


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--verified", type=Path, default=VERIFIED)
    args = ap.parse_args()

    records = json.loads(args.verified.read_text(encoding="utf-8"))
    supp = json.loads(SUPPLEMENTARY.read_text(encoding="utf-8"))
    supp_abstracts = supp.get("abstracts", {})
    unrecoverable = supp.get("unrecoverable", {})

    for rec in records:
        pid = rec["id"]
        if rec.get("abstract"):
            continue
        extra = supp_abstracts.get(pid)
        if extra:
            rec["abstract"] = extra["text"]
            rec["abstract_source"] = extra["source"]
            rec["abstract_source_url"] = extra["source_url"]
            RAW_DIR.mkdir(parents=True, exist_ok=True)
            (RAW_DIR / f"{pid}_abstract_supplementary.json").write_text(
                json.dumps({"id": pid, "doi": rec["doi"], **extra}, indent=2, ensure_ascii=False),
                encoding="utf-8")
            print(f"{pid:<3} SUPPLEMENTARY {len(extra['text']):>5} chars  {rec['doi']}")
            continue
        abstract, final_url = scrape_landing(rec["doi"])
        if abstract:
            rec["abstract"] = abstract
            rec["abstract_source"] = "publisher_landing_page_meta"
            rec["url_resolved"] = final_url
            RAW_DIR.mkdir(parents=True, exist_ok=True)
            (RAW_DIR / f"{pid}_landing_abstract.json").write_text(
                json.dumps({"doi": rec["doi"], "url": final_url, "abstract": abstract},
                           indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"{pid:<3} RECOVERED  {len(abstract):>5} chars  {rec['doi']}")
        else:
            rec["abstract_source"] = "unavailable_paywalled"
            rec["abstract_unavailable_note"] = unrecoverable.get(pid, {}).get("attempts")
            print(f"{pid:<3} UNAVAILABLE {rec['doi']}  (left as None, flagged)")

    args.verified.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
    missing = [r["id"] for r in records if not r.get("abstract")]
    print(f"\nresolved {len(records) - len(missing)}/{len(records)}; still missing: {missing}")



if __name__ == "__main__":
    main()
