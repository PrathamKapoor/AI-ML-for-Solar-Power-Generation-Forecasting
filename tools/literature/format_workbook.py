"""Apply the workbook formatting the audit requires.

    python tools/literature/format_workbook.py

Fixes only what ``audit_workbook_formatting.py`` reports, so the two stay in step:

* freeze the header row and add an autofilter across the used range on every
  sheet, so a reader can sort and scroll without losing the column names;
* raise any column narrower than the minimum, which otherwise renders numbers as
  ``###``;
* attach a real hyperlink to every DOI and paper-link cell, resolving a bare DOI
  to ``https://doi.org/<doi>`` and leaving an existing URL alone;
* bold the header row, and drop trailing columns that have neither a header nor
  any value, which otherwise read as a column with no name.

No cell value is changed. The script is idempotent, so it can be re-run after any
regeneration of the workbook.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WORKBOOK = ROOT / "literature" / "literature_review.xlsx"

MIN_WIDTH = 10
MAX_WIDTH = 90
IDENTIFIER_HEADERS = {"doi", "paper link", "paper link (doi)"}
DOI_PATTERN = re.compile(r"^10\.\d{4,9}/\S+$")


def hyperlink_for(text: str) -> str | None:
    """A resolvable URL for a cell holding a DOI or a link, or None."""
    value = str(text).strip()
    if not value:
        return None
    if value.startswith("http://") or value.startswith("https://"):
        return value
    if DOI_PATTERN.match(value):
        return f"https://doi.org/{value}"
    if value.startswith("10."):
        return f"https://doi.org/{value}"
    return None


def main() -> int:
    try:
        from openpyxl import load_workbook
        from openpyxl.styles import Alignment, Font
    except ImportError:
        print("openpyxl is not installed; cannot format the workbook", file=sys.stderr)
        return 1
    if not WORKBOOK.exists():
        print(f"missing {WORKBOOK}", file=sys.stderr)
        return 1

    workbook = load_workbook(WORKBOOK)
    frozen = filtered = widened = linked = 0
    dropped = 0

    for sheet in workbook.worksheets:
        if sheet.max_row < 1 or sheet.max_column < 1:
            continue
        last = sheet.cell(row=sheet.max_row, column=sheet.max_column).coordinate

        if sheet.freeze_panes != "A2":
            sheet.freeze_panes = "A2"
            frozen += 1

        ref = f"A1:{last}"
        if sheet.auto_filter.ref != ref:
            sheet.auto_filter.ref = ref
            filtered += 1

        header = sheet.cell(row=1, column=1)
        header.font = Font(bold=True)
        header.alignment = Alignment(vertical="top", wrap_text=True)

        for index in range(1, sheet.max_column + 1):
            cell = sheet.cell(row=1, column=index)
            letter = cell.column_letter
            dimension = sheet.column_dimensions.get(letter)
            width = dimension.width if dimension is not None else None
            if width is None or width < MIN_WIDTH:
                sheet.column_dimensions[letter].width = MIN_WIDTH
                widened += 1
            elif width > MAX_WIDTH:
                sheet.column_dimensions[letter].width = MAX_WIDTH

        # A trailing column with no header and no values is not part of the sheet.
        # It is dropped rather than given a placeholder name, because a column
        # called "(unused)" invites a reader to wonder what should have been in it,
        # whereas an absent column says plainly that there is nothing there.
        while sheet.max_column > 1 and all(
                sheet.cell(row=row, column=sheet.max_column).value in (None, "")
                for row in range(1, sheet.max_row + 1)):
            sheet.delete_cols(sheet.max_column)
            dropped += 1

        headers = {str(c.value).strip().lower(): c.column
                   for c in sheet[1] if c.value}
        identifier_column = next((c for h, c in headers.items()
                                  if h in IDENTIFIER_HEADERS), None)
        if identifier_column is None:
            continue
        for row in range(2, sheet.max_row + 1):
            cell = sheet.cell(row=row, column=identifier_column)
            if cell.value is None or not str(cell.value).strip():
                continue
            if cell.hyperlink is not None:
                continue
            url = hyperlink_for(cell.value)
            if url is None:
                continue
            cell.hyperlink = url
            cell.style = "Hyperlink"
            linked += 1

    workbook.save(WORKBOOK)
    print(f"sheets formatted    : {len(workbook.worksheets)}")
    print(f"headers frozen      : {frozen}")
    print(f"autofilters applied : {filtered}")
    print(f"columns widened     : {widened}")
    print(f"hyperlinks attached : {linked}")
    print(f"empty columns dropped: {dropped}")
    print(f"written             : {WORKBOOK}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



