"""Summarise the literature-review workbook for inspection.

Prints the category distribution, the year spread and the research-gap sheet.
Read-only: it makes no changes to the workbook.

Run:  python tools/literature/summarise_workbook.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
WORKBOOK = ROOT / "literature" / "literature_review.xlsx"


def main() -> int:
    analysis = pd.read_excel(WORKBOOK, sheet_name="Literature Analysis")
    print(f"papers: {len(analysis)}")
    print("\n=== model family distribution ===")
    print(analysis["Model Family"].value_counts().to_string())
    print("\n=== publication years ===")
    years = sorted(int(str(value)[:4]) for value in analysis["Year"]
                   if str(value)[:4].isdigit())
    print(f"range {min(years)}-{max(years)}, n={len(years)}")
    print("\n=== horizons covered ===")
    for value in analysis["Forecast Horizon"].dropna().unique():
        print(f"  {value}")
    print("\n=== search scope ===")
    scope = pd.read_excel(WORKBOOK, sheet_name="Selection and Exclusions")
    for _, row in scope.iterrows():
        text = str(row.iloc[1])
        if text != "nan":
            print(f"  {str(row.iloc[0])[:34]:34} {text[:96]}")

    gaps = pd.read_excel(WORKBOOK, sheet_name="Research Gaps")
    print(f"\n=== research gaps ({len(gaps)}) ===")
    for _, row in gaps.iterrows():
        print(f"\n{row['Gap ID']}: {str(row['Research Gap'])[:96]}")
        print(f"  evidence: {str(row['Evidence Papers'])[:110]}")
        print(f"  RQ: {str(row['Testable Research Question'])[:110]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
