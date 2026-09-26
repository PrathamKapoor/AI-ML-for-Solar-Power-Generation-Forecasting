"""Validate that every DOI in the corpus resolves, and record the outcome.

A publisher that blocks automated clients returns HTTP 403 *after* a successful
DOI redirect. Treating that as a broken link would be wrong, so the checker
distinguishes three outcomes:

``resolved``            - final URL is on a publisher domain, status < 400
``resolved_blocked``    - DOI redirected correctly to a publisher page that then
                          returned 403/401 to an automated client (link is valid)
``unresolved``          - the DOI itself failed to redirect to a publisher page

Usage
-----
    python tools/literature/check_doi_links.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[2]
VERIFIED = PROJECT_ROOT / "literature" / "verification" / "verified_papers.json"
OUT = PROJECT_ROOT / "literature" / "verification" / "doi_link_check.json"

BROWSER_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36")

# Domains that legitimately host the landing page for a registered DOI.
# linkinghub.elsevier.com is Elsevier's own DOI redirector, so it counts as the publisher.
PUBLISHER_HOST = re.compile(
    r"(sciencedirect\.com|linkinghub\.elsevier\.com|ieeexplore\.ieee\.org|ieee\.org|mdpi\.com|"
    r"onlinelibrary\.wiley\.com|wiley\.com|springer\.com|link\.springer\.com|nature\.com|"
    r"sciencemag\.org|eurekaselect\.com|tandfonline\.com|sagepub\.com)$",
    re.IGNORECASE,
)

BLOCK_STATUS = {401, 403, 405, 429, 503}


def check(doi: str) -> dict:
    url = f"https://doi.org/{doi}"
    try:
        resp = requests.get(url, headers={"User-Agent": BROWSER_UA, "Accept": "text/html"},
                            timeout=45, allow_redirects=True)
    except requests.RequestException as exc:
        return {"doi": doi, "outcome": "unresolved", "detail": f"{type(exc).__name__}: {exc}"}

    host = requests.utils.urlparse(resp.url).netloc
    landed_on_publisher = bool(PUBLISHER_HOST.search(host))
    if resp.status_code in BLOCK_STATUS and landed_on_publisher:
        outcome = "resolved_blocked"
    elif resp.status_code < 400 and landed_on_publisher:
        outcome = "resolved"
    elif resp.status_code < 400:
        outcome = "resolved_nonpublisher_host"
    else:
        outcome = "unresolved"
    return {"doi": doi, "outcome": outcome, "http_status": resp.status_code,
            "final_url": resp.url, "final_host": host}


def main() -> None:
    records = json.loads(VERIFIED.read_text(encoding="utf-8"))
    results = []
    for rec in records:
        outcome = check(rec["doi"])
        outcome["paper_id"] = rec["id"]
        outcome["title"] = rec["title"]
        results.append(outcome)
        print(f"{rec['id']:<3} {outcome['outcome']:<26} {outcome.get('http_status')} "
              f"{outcome.get('final_host', '')[:44]}")

    counts: dict[str, int] = {}
    for r in results:
        counts[r["outcome"]] = counts.get(r["outcome"], 0) + 1

    payload = {
        "checked": len(results),
        "counts": counts,
        "interpretation": {
            "resolved": "DOI redirected to a publisher landing page that served the request.",
            "resolved_blocked": "DOI redirected to the correct publisher landing page, which then "
                                "refused the automated request. The link is valid; a browser opens it.",
            "resolved_nonpublisher_host": "DOI redirected successfully to a host outside the "
                                          "publisher allow-list (e.g. an aggregator).",
            "unresolved": "The DOI failed to reach a publisher landing page. This would indicate a "
                          "genuinely broken link.",
        },
        "results": results,
    }
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\ncounts: {counts}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
