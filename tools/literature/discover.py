"""Category-targeted literature discovery.

The seed list does not reliably populate every required category (classical ML,
transformer-family architectures, explainable/probabilistic PV forecasting).
This module issues explicit topical queries against Crossref and OpenAlex and
writes ranked candidates with verified metadata so a human/agent can select
genuine peer-reviewed replacements.

No paper is auto-selected; the output is a candidate pool with provenance.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from resolve_seeds import (OPENALEX, PROJECT_ROOT, query_crossref, query_openalex,
                           title_similarity, crossref_to_record, openalex_to_record)

OUT_DIR = PROJECT_ROOT / "literature" / "verification"

QUERIES: dict[str, list[str]] = {
    "A_classical_ml": [
        "support vector regression photovoltaic power forecasting",
        "random forest photovoltaic power generation forecasting",
        "XGBoost short-term photovoltaic power forecasting",
        "gradient boosting solar power forecasting comparative machine learning",
        "linear regression solar irradiance nowcasting persistence baseline",
        "photovoltaic power forecasting machine learning benchmark comparison",
    ],
    "D_advanced_architectures": [
        "transformer photovoltaic power forecasting",
        "iTransformer photovoltaic power forecasting",
        "PatchTST solar power forecasting",
        "TimesNet photovoltaic power forecasting",
        "spatial-temporal graph neural network multi-site photovoltaic power forecasting",
        "informer autoformer solar power forecasting",
    ],
    "F_explainability_uncertainty": [
        "explainable machine learning solar power forecasting SHAP",
        "interpretable photovoltaic power forecasting feature importance",
        "probabilistic photovoltaic power forecasting prediction intervals deep learning",
        "quantile regression solar power forecasting uncertainty",
        "explainable AI photovoltaic power forecasting XGBoost SHAP",
    ],
    "B_rnn": [
        "GRU solar power forecasting",
        "bidirectional LSTM photovoltaic power forecasting",
        "LSTM solar irradiance forecasting comparison",
    ],
    "E_physics_informed": [
        "physics-informed neural network photovoltaic power forecasting",
        "solar position features photovoltaic power forecasting domain knowledge",
        "hybrid physical statistical machine learning solar power forecasting",
    ],
}

# Filters applied to keep the pool at peer-review quality.
EXCLUDE_TYPES = {"posted-content", "component", "dataset", "reference-entry", "proceedings-article"}


def discover(category: str, query: str, rows: int = 10) -> list[dict]:
    pool: dict[str, dict] = {}

    for item in query_crossref(query, rows=rows):
        rec = crossref_to_record(item, 0.0)
        if rec["type"] in EXCLUDE_TYPES or not rec["title"] or not rec["doi"]:
            continue
        pool.setdefault(rec["doi"].lower(), rec)

    for item in query_openalex(query, per_page=rows):
        rec = openalex_to_record(item, 0.0)
        if rec["type"] in {"posted-content", "dataset"} or not rec["title"]:
            continue
        key = (rec["doi"] or rec["title"]).lower()
        pool.setdefault(key, rec)

    # Rank by topical relevance to the category intent.
    terms = {t for t in query.lower().split() if len(t) > 3}
    for rec in pool.values():
        blob = f"{rec.get('title') or ''} {(rec.get('container') or '')}".lower()
        overlap = sum(1 for t in terms if t in blob)
        rec["relevance_hits"] = overlap
        rec["query"] = query
        rec["peer_reviewed_hint"] = rec.get("source") != "openalex" or rec.get("type") in {
            "article", "review", "journal-article", "proceedings-article"}
    ranked = sorted(pool.values(), key=lambda r: (r["relevance_hits"], r.get("cited_by_count") or 0), reverse=True)
    return ranked[:12]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--category", default=None, help="restrict to one category key")
    ap.add_argument("--out", type=Path, default=OUT_DIR / "discovery_candidates.json")
    args = ap.parse_args()

    categories = {args.category: QUERIES[args.category]} if args.category else QUERIES
    result: dict[str, list[dict]] = {}
    for cat, queries in categories.items():
        result[cat] = []
        seen: set[str] = set()
        for q in queries:
            for rec in discover(cat, q):
                key = (rec.get("doi") or rec.get("title") or "").lower()
                if key in seen:
                    continue
                seen.add(key)
                rec["category_seed"] = cat
                result[cat].append(rec)
            print(f"  [{cat}] {q[:60]:<60} pool={len(result[cat])}")
            time.sleep(0.35)

    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    for cat, recs in result.items():
        print(f"\n### {cat}  ({len(recs)} candidates)")
        for r in recs[:10]:
            print(f"  hits={r['relevance_hits']} cites={r.get('cited_by_count')} "
                  f"{(r.get('title') or '')[:82]}")
            print(f"      {r.get('container')} | {r.get('doi') or r.get('url')}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
