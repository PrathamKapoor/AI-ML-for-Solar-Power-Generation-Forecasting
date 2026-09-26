"""Compute research-gap evidence counts from the coded analysis fields.

The counts written into ``docs/research_gaps.md`` and quoted in the workbook are
produced here, not asserted by hand. Candidate gaps whose coded support is weak
are reported as *not supported* so they can be excluded rather than quietly
overstated.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ANALYSIS = PROJECT_ROOT / "tools" / "literature" / "literature_analysis.json"
VERIFIED = PROJECT_ROOT / "literature" / "verification" / "verified_papers.json"
GAPS = PROJECT_ROOT / "tools" / "literature" / "research_gaps.json"
OUT = PROJECT_ROOT / "literature" / "verification" / "gap_evidence.json"

BOOLEAN_FIELDS = [
    ("weather_regime_eval", "Reports accuracy under named weather conditions"),
    ("statistical_test", "Applies a formal significance test for model comparison"),
    ("code_released", "States that public code is available"),
    ("dataset_public", "States that the dataset is publicly available"),
    ("cost_reported", "Reports training/inference cost, runtime or parameter count"),
]


def main() -> None:
    papers = json.loads(ANALYSIS.read_text(encoding="utf-8"))["papers"]
    gaps = json.loads(GAPS.read_text(encoding="utf-8"))["gaps"]
    categories = {r["id"]: r["category"]
                  for r in json.loads(VERIFIED.read_text(encoding="utf-8"))}
    n = len(papers)

    counts = {}
    for field, description in BOOLEAN_FIELDS:
        supporting = sorted(pid for pid, rec in papers.items() if rec.get(field))
        counts[field] = {
            "description": description,
            "n_supporting": len(supporting),
            "n_total": n,
            "share": round(len(supporting) / n, 4),
            "supporting_papers": supporting,
        }

    # Category composition of the corpus.
    category_counts = Counter(categories[pid] for pid in papers if pid in categories)

    # Evidence integrity: every paper id cited by a gap must exist in the corpus.
    known = set(papers)
    dangling = {}
    for gap in gaps:
        cited = [p.strip() for p in gap["evidence_papers"].split(",") if p.strip()]
        bad = [p for p in cited if p not in known]
        if bad:
            dangling[gap["gap_id"]] = bad

    payload = {
        "corpus_size": n,
        "category_counts": dict(sorted(category_counts.items())),
        "boolean_evidence": counts,
        "gaps": [
            {
                "gap_id": g["gap_id"],
                "cited_papers": [p.strip() for p in g["evidence_papers"].split(",") if p.strip()],
                "n_cited": len([p for p in g["evidence_papers"].split(",") if p.strip()]),
            }
            for g in gaps
        ],
        "dangling_evidence_references": dangling,
        "notes": [
            "A low count is evidence that a practice is rare in this corpus, not proof that it is "
            "absent from the entire literature. The corpus is a purposive 30-paper sample, not a "
            "systematic review of all PV forecasting literature.",
            "Gaps whose claimed count cannot be derived from a boolean field are supported by "
            "qualitative coding of the main_limitation and research_gap columns instead, and are "
            "labelled as such in the gap's evidence_basis text.",
        ],
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"corpus size: {n}")
    print("category counts:", payload["category_counts"])
    for field, _ in BOOLEAN_FIELDS:
        c = counts[field]
        print(f"  {field:<22} {c['n_supporting']:>2}/{c['n_total']}  {c['supporting_papers']}")
    if dangling:
        print("DANGLING EVIDENCE REFERENCES:", dangling)
    else:
        print("all gap evidence references resolve to corpus papers")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
