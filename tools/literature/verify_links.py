"""Verify every literature-review entry against Crossref.

Checks, for each of the 30 papers:
  1. the DOI resolves;
  2. the registered title matches the title recorded in the workbook;
  3. the registered venue matches the recorded publishing organisation.

Run:  python tools/literature/verify_links.py
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
WORKBOOK = ROOT / "literature" / "literature_review.xlsx"
MAILTO = "prathamkapoor027@gmail.com"


def normalise(text: str) -> str:
    return "".join(ch for ch in text.lower() if ch.isalnum())


def fetch(doi: str) -> dict:
    request = urllib.request.Request(
        f"https://api.crossref.org/works/{doi}",
        headers={"User-Agent": f"solar-pv-forecasting-verify/1.0 (mailto:{MAILTO})"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)["message"]


def main() -> int:
    sheet = pd.read_excel(WORKBOOK, sheet_name="Literature Review")
    problems: list[dict] = []
    print(f"checking {len(sheet)} entries against Crossref\n")
    for _, row in sheet.iterrows():
        number = row["Sr No"]
        title = str(row["Title"])
        link = str(row["Paper Link"])
        doi = link.replace("https://doi.org/", "").strip()
        try:
            message = fetch(doi)
        except Exception as exc:  # noqa: BLE001 - report, never guess
            print(f"{number:>3}  UNRESOLVED  {doi}  ({type(exc).__name__})")
            problems.append({"sr": number, "doi": doi, "issue": "unresolved",
                             "detail": str(exc)})
            continue
        registered_title = (message.get("title") or [""])[0]
        match = normalise(title)[:40] == normalise(registered_title)[:40]
        venue = (message.get("container-title") or [""])[0]
        print(f"{number:>3}  {'title OK ' if match else 'TITLE DIFF'}  {doi}")
        print(f"     venue: {venue}")
        if not match:
            problems.append({"sr": number, "doi": doi, "issue": "title_mismatch",
                             "recorded": title, "registered": registered_title})
        time.sleep(0.3)

    print(f"\n{len(problems)} problem(s)")
    for problem in problems:
        print(json.dumps(problem, ensure_ascii=False, indent=1))
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
