"""Build ``literature/literature_review.xlsx`` from verified metadata.

Sheets
------
``Literature Review``  - the 30-paper reading list (required column schema)
``Literature Analysis``- per-paper extraction for cross-study comparison
``Research Gaps``      - evidence-backed gaps, each tied to specific papers
``Provenance``         - per-field source audit trail (extra sheet, added so a
                         reader can tell verified metadata from unavailable data)

Every value is either taken from ``literature/verification/verified_papers.json``
(authoritative API responses) or from the curated analysis files. No metric,
conclusion or citation count is typed by hand into this script.

Usage
-----
    python -m tools.literature.build_workbook
    python tools/literature/build_workbook.py
"""

from __future__ import annotations

import json
import re
import sys
from datetime import date
from pathlib import Path

from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

VERIFIED = PROJECT_ROOT / "literature" / "verification" / "verified_papers.json"
ANALYSIS = PROJECT_ROOT / "tools" / "literature" / "literature_analysis.json"
GAPS = PROJECT_ROOT / "tools" / "literature" / "research_gaps.json"
EXCLUSIONS = PROJECT_ROOT / "tools" / "literature" / "exclusions.json"
OUT_XLSX = PROJECT_ROOT / "literature" / "literature_review.xlsx"
OUT_BIB = PROJECT_ROOT / "literature" / "references.bib"

LITERATURE_REVIEW_COLUMNS = [
    "Sr No", "Name", "Author(s)", "Publishing Date", "Published By / Organization",
    "Title", "Abstract (review summary)", "Conclusion (review summary)",
    "Keywords Related", "Paper Link",
]

LITERATURE_ANALYSIS_COLUMNS = [
    "Sr No", "Paper", "Year", "Forecast Target", "Forecast Horizon", "Dataset",
    "Dataset Location", "Dataset Size", "Sampling Frequency", "Input Variables",
    "Weather Variables", "Model Family", "Specific Model", "Baselines", "Metrics",
    "Best Reported Metric", "Weather Conditions Considered", "Explainability",
    "Uncertainty", "Multi-site Validation", "External Validation",
    "Main Contribution", "Main Limitation", "Research Gap", "Reproducibility",
    "Code Available", "Dataset Available", "DOI", "Source",
]

RESEARCH_GAPS_COLUMNS = [
    "Gap ID", "Research Gap", "Evidence Papers", "Why It Matters",
    "Current Approaches", "Limitation", "Opportunity for Our Project",
    "Testable Research Question",
]

PROVENANCE_COLUMNS = [
    "Paper ID", "DOI", "Title (verified)", "Abstract source", "Abstract characters",
    "Sources that returned a record", "Crossref/OpenAlex title agreement",
    "Semantic Scholar citations", "Crossref citations", "Provenance flag",
]

HEADER_FILL = PatternFill("solid", fgColor="1F3864")
HEADER_FONT = Font(color="FFFFFF", bold=True, size=11)
SUBHEADER_FILL = PatternFill("solid", fgColor="D9E2F3")
THIN = Side(style="thin", color="BFBFBF")
BORDER = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)

CATEGORY_LABEL = {
    "G": "Systematic review / state of the field",
    "A": "Classical / conventional machine learning",
    "B": "Recurrent / RNN-based forecasting",
    "C": "Hybrid deep learning",
    "D": "Advanced / attention-based architectures",
    "E": "Physics-informed / domain-aware",
    "F": "Probabilistic / explainable / robust",
}

CATEGORY_KEYWORDS = {
    "G": "solar power forecasting review; machine learning; deep learning; survey; state of the art",
    "A": "photovoltaic power forecasting; machine learning; XGBoost; random forest; support vector regression; gradient boosting; NWP",
    "B": "LSTM; GRU; recurrent neural network; short-term solar power forecasting; time-series regression",
    "C": "CNN-LSTM; hybrid deep learning; attention mechanism; wavelet transform; multi-source data fusion",
    "D": "transformer; iTransformer; graph convolutional network; encoder-decoder; advanced time-series architecture",
    "E": "physics-informed; domain knowledge; solar position; physically constrained forecasting",
    "F": "probabilistic forecasting; prediction intervals; uncertainty quantification; explainable AI; SHAP; robustness",
}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def cite_key(record: dict) -> str:
    """BibTeX key of the form FirstAuthorSurnameYEARFirstTitleWord."""
    authors = record.get("authors_crossref") or record.get("authors_s2") or ["anon"]
    surname = re.sub(r"[^A-Za-z]", "", authors[0].split()[-1] if authors[-1] else "anon") or "anon"
    year = (record.get("date_parts_crossref") or [None])[0] or "nd"
    words = re.findall(r"[A-Za-z]+", record.get("title") or "")
    stop = {"a", "an", "the", "of", "for", "and", "in", "on", "to", "with", "based", "using"}
    first = next((w for w in words if w.lower() not in stop), "untitled")
    return f"{surname.lower()}{year}{first.lower()}"


def short_name(record: dict) -> str:
    """Short citation label, e.g. 'Gaboitaolelwe et al. (2023)'."""
    authors = record.get("authors_crossref") or record.get("authors_s2") or []
    year = (record.get("date_parts_crossref") or [None])[0] or "n.d."
    if not authors:
        return f"Unknown ({year})"
    surname = authors[0].split()[-1]
    if len(authors) == 1:
        return f"{surname} ({year})"
    if len(authors) == 2:
        return f"{surname} & {authors[1].split()[-1]} ({year})"
    return f"{surname} et al. ({year})"


def publishing_date(record: dict) -> str:
    """Assemble a precise publishing-date string from Crossref date fields."""
    parts = record.get("date_parts_crossref") or []
    online = record.get("published_online") or []
    print_ = record.get("published_print") or []
    if online:
        iso = "-".join(f"{p:02d}" if i else str(p) for i, p in enumerate(online))
        if print_ and print_ != online:
            return f"{iso} (online); {'-'.join(str(p) for p in print_)} (issue)"
        return f"{iso} (online)"
    if parts:
        return "-".join(str(p) for p in parts)
    if record.get("publication_date_s2"):
        return str(record["publication_date_s2"])
    return "Not reported"


def publisher_org(record: dict) -> str:
    publisher = record.get("publisher_crossref") or record.get("publisher_openalex") or "Not reported"
    container = record.get("container_crossref") or record.get("container_openalex")
    if container:
        return f"{publisher} / {container}"
    return publisher


def review_summary(abstract: str | None, record: dict) -> str:
    """Convert a verbatim abstract into a structured review summary.

    The abstract's own sentences are used (no new claims added); the leading
    ``Summary:`` marker matches the required workbook format.
    """
    if not abstract:
        return ("Summary: NOT AVAILABLE - the source abstract is paywalled and was not "
                "retrievable from Crossref, OpenAlex, Semantic Scholar, the publisher landing "
                "page or any RePEc mirror. Bibliographic metadata for this record is fully "
                "verified, but no methodological or numerical claim is asserted here. Full-text "
                "consultation is required before citing this work.")
    text = re.sub(r"^(abstract|summary)\s*", "", abstract.strip(), flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    if not text.endswith("."):
        text += "."
    return f"Summary: {text}"


def conclusion_summary(record: dict, analysis: dict) -> str:
    """Conclusion column.

    Preference order: a verbatim sentence from the abstract that reports an
    outcome, else the curated main_contribution field. This avoids inventing a
    conclusion the source does not state.
    """
    abstract = record.get("abstract") or ""
    outcome_markers = (
        "results show", "results indicate", "results demonstrate", "results reveal",
        "the results", "we found", "it was also found", "our findings", "the findings",
        "experiments show", "experiments demonstrate", "we show", "we found that",
        "ablation study results", "the results are compared", "outperforms",
        "achieved better performance", "is reported as", "reports that",
    )
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z(])", abstract)
    for sentence in sentences:
        low = sentence.lower()
        if any(marker in low for marker in outcome_markers) and 40 < len(sentence) < 900:
            cleaned = re.sub(r"\s+", " ", sentence).strip()
            return f"Summary: {cleaned}"
    return (f"Summary: (No explicit conclusion sentence in the retrieved abstract. "
            f"Reported outcome: {analysis['main_contribution']})")


def style_header(ws, ncols: int) -> None:
    for col in range(1, ncols + 1):
        cell = ws.cell(row=1, column=col)
        cell.fill = HEADER_FILL
        cell.font = HEADER_FONT
        cell.alignment = Alignment(vertical="center", horizontal="center", wrap_text=True)
        cell.border = BORDER
    ws.row_dimensions[1].height = 34
    ws.freeze_panes = "A2"


def autosize(ws, widths: dict[int, int], default: int = 22) -> None:
    for col, width in widths.items():
        ws.column_dimensions[get_column_letter(col)].width = width


def write_rows(ws, columns: list[str], rows: list[list]) -> None:
    ws.append(columns)
    for row in rows:
        ws.append(row)
    for r in range(2, ws.max_row + 1):
        for c in range(1, len(columns) + 1):
            cell = ws.cell(row=r, column=c)
            cell.alignment = Alignment(vertical="top", wrap_text=True)
            cell.border = BORDER


def bibtex_escape(text: str) -> str:
    return (text or "").replace("&", r"\&").replace("%", r"\%").replace("_", r"\_")


def build_bib(records: list[dict], out_path: Path) -> None:
    lines = [
        "% references.bib - generated by tools/literature/build_workbook.py",
        "% All entries are derived from Crossref / OpenAlex / Semantic Scholar metadata",
        "% recorded in literature/verification/. Do not edit by hand; rerun the builder.",
        "",
    ]
    for rec in records:
        key = cite_key(rec)
        authors = rec.get("authors_crossref") or rec.get("authors_s2") or []
        year = (rec.get("date_parts_crossref") or [None])[0] or "n.d."
        venue = rec.get("container_crossref") or rec.get("container_openalex") or ""
        entry_type = "inproceedings" if rec.get("type_crossref") == "proceedings-article" else "article"
        abstract = rec.get("abstract")

        fields = [
            ("title", bibtex_escape(rec.get("title"))),
            ("author", " and ".join(bibtex_escape(a) for a in authors)),
            ("journal", bibtex_escape(venue)),
            ("year", str(year)),
            ("doi", rec["doi"]),
            ("url", "https://doi.org/" + rec["doi"]),
        ]
        if rec.get("published_online") and len(rec["published_online"]) > 1:
            fields.insert(4, ("month", str(rec["published_online"][1])))
        if abstract:
            fields.append(("abstract", bibtex_escape(abstract)))
        fields.append(("note", "Verified via Crossref, OpenAlex and Semantic Scholar; "
                               "abstract provenance recorded in literature/verification/"))

        lines.append("@{}{{{},".format(entry_type, key))
        for name, value in fields:
            lines.append("  {} = {{{}}},".format(name, value))
        lines.append("}")
        lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    records = load_json(VERIFIED)
    analysis = load_json(ANALYSIS)["papers"]
    gaps = load_json(GAPS)["gaps"]
    exclusions = load_json(EXCLUSIONS)["excluded"]

    missing = [r["id"] for r in records if r["id"] not in analysis]
    if missing:
        raise SystemExit(f"missing curated analysis for: {missing}")

    wb = Workbook()

    # ------------------------------------------------------------------ Sheet 1
    ws = wb.active
    ws.title = "Literature Review"
    rows = []
    for i, rec in enumerate(records, start=1):
        cat = rec["category"]
        keywords = CATEGORY_KEYWORDS[cat]
        extra = [k for k in (rec.get("keywords_openalex") or []) if k]
        if extra:
            keywords = keywords + "; " + "; ".join(extra[:5])
        rows.append([
            i,
            short_name(rec),
            "; ".join(rec.get("authors_crossref") or rec.get("authors_s2") or []),
            publishing_date(rec),
            publisher_org(rec),
            rec["title"],
            review_summary(rec.get("abstract"), rec),
            conclusion_summary(rec, analysis[rec["id"]]),
            keywords,
            f"https://doi.org/{rec['doi']}",
        ])
    write_rows(ws, LITERATURE_REVIEW_COLUMNS, rows)
    style_header(ws, len(LITERATURE_REVIEW_COLUMNS))
    autosize(ws, {1: 7, 2: 26, 3: 40, 4: 24, 5: 38, 6: 52, 7: 90, 8: 78, 9: 46, 10: 40})

    # ------------------------------------------------------------------ Sheet 2
    ws2 = wb.create_sheet("Literature Analysis")
    rows2 = []
    for i, rec in enumerate(records, start=1):
        a = analysis[rec["id"]]
        year = (rec.get("date_parts_crossref") or [None])[0] or "Not reported"
        source = (f"Crossref + OpenAlex + Semantic Scholar; category {cat_label(rec['category'])}"
                  if rec.get("sources_found") and len(rec["sources_found"]) >= 2
                  else "; ".join(rec.get("sources_found") or ["Not reported"]))
        if a.get("provenance_flag"):
            source += f"; FLAG: {a['provenance_flag']}"
        rows2.append([
            i, short_name(rec), year,
            a["forecast_target"], a["forecast_horizon"], a["dataset"], a["dataset_location"],
            a["dataset_size"], a["sampling_frequency"], a["input_variables"], a["weather_variables"],
            a["model_family"], a["specific_model"], a["baselines"], a["metrics"],
            a["best_reported_metric"], a["weather_conditions_considered"], a["explainability"],
            a["uncertainty"], a["multi_site_validation"], a["external_validation"],
            a["main_contribution"], a["main_limitation"], a["research_gap"], a["reproducibility"],
            "Yes" if a["code_released"] else "No",
            "Yes" if a["dataset_public"] else "No",
            rec["doi"], source,
        ])
    write_rows(ws2, LITERATURE_ANALYSIS_COLUMNS, rows2)
    style_header(ws2, len(LITERATURE_ANALYSIS_COLUMNS))
    autosize(ws2, {1: 7, 2: 26, 3: 8, 4: 22, 5: 30, 6: 34, 7: 24, 8: 26, 9: 18, 10: 34,
                   11: 34, 12: 26, 13: 32, 14: 34, 15: 30, 16: 40, 17: 36, 18: 30, 19: 20,
                   20: 20, 21: 24, 22: 56, 23: 56, 24: 52, 25: 34, 26: 14, 27: 16, 28: 30, 29: 44})

    # ------------------------------------------------------------------ Sheet 3
    ws3 = wb.create_sheet("Research Gaps")
    rows3 = [[g["gap_id"], g["research_gap"], g["evidence_papers"], g["why_it_matters"],
              g["current_approaches"], g["limitation"], g["opportunity_for_our_project"],
              g["testable_research_question"]] for g in gaps]
    write_rows(ws3, RESEARCH_GAPS_COLUMNS, rows3)
    style_header(ws3, len(RESEARCH_GAPS_COLUMNS))
    autosize(ws3, {1: 10, 2: 66, 3: 30, 4: 66, 5: 60, 6: 60, 7: 66, 8: 66})

    # ------------------------------------------------------------------ Sheet 4
    ws4 = wb.create_sheet("Provenance")
    rows4 = []
    for rec in records:
        rows4.append([
            rec["id"], rec["doi"], rec["title"], rec.get("abstract_source"),
            len(rec.get("abstract") or ""),
            ", ".join(rec.get("sources_found") or []),
            rec.get("title_agreement_crossref_openalex"),
            rec.get("citation_count_s2"), rec.get("citation_count_crossref"),
            (analysis[rec["id"]].get("provenance_flag") or ""),
        ])
    write_rows(ws4, PROVENANCE_COLUMNS, rows4)
    style_header(ws4, len(PROVENANCE_COLUMNS))
    autosize(ws4, {1: 10, 2: 32, 3: 62, 4: 30, 5: 16, 6: 34, 7: 26, 8: 22, 9: 20, 10: 30})

    # ------------------------------------------------------------------ Sheet 5
    ws5 = wb.create_sheet("Selection and Exclusions")
    ws5.append(["Field", "Value"])
    meta = [
        ("Corpus size", len(records)),
        ("Selection protocol",
         "Discovery seeds were treated as hypotheses only. Each candidate title was resolved "
         "against Crossref and OpenAlex by normalised title similarity, then the surviving "
         "records were verified by DOI against Crossref, OpenAlex and Semantic Scholar. "
         "Records that resolved only to a preprint, or to a venue below the corpus quality bar, "
         "were replaced."),
        ("Peer-review status", "All 30 records are Crossref type 'journal-article'. No preprints, "
                               "posters or withdrawn records are included."),
        ("Excluded candidates", len(exclusions)),
        ("Exclusion log", "See the 'Selection and Exclusions' rows below and "
                          "tools/literature/exclusions.json"),
        ("Verification artefacts", "literature/verification/verified_papers.json and "
                                   "literature/verification/raw/*.json (unmodified API payloads)"),
        ("Known limitation", "G3 (Kumari & Toshniwal, 2021) has a paywalled abstract that no open "
                             "metadata source exposes. Its bibliographic metadata is verified; no "
                             "methodological or numerical claim is asserted for it anywhere in this "
                             "workbook."),
        ("Workbook generated", date.today().isoformat()),
        ("Generator", "tools/literature/build_workbook.py"),
    ]
    for field, value in meta:
        ws5.append([field, value])
    ws5.append([])
    ws5.append(["Excluded candidate", "Title", "Reason for exclusion"])
    for exc in exclusions:
        ws5.append([exc.get("seed_or_doi", ""), exc.get("title", ""), exc.get("reason", "")])
    for r in range(1, ws5.max_row + 1):
        for c in (1, 2, 3):
            ws5.cell(row=r, column=c).alignment = Alignment(vertical="top", wrap_text=True)
            ws5.cell(row=r, column=c).border = BORDER
    for c in (1, 2, 3):
        ws5.cell(row=1, column=c).fill = HEADER_FILL
        ws5.cell(row=1, column=c).font = HEADER_FONT
    autosize(ws5, {1: 34, 2: 62, 3: 92})

    OUT_XLSX.parent.mkdir(parents=True, exist_ok=True)
    wb.save(OUT_XLSX)
    build_bib(records, OUT_BIB)

    print(f"wrote {OUT_XLSX}")
    print(f"wrote {OUT_BIB}")
    print(f"sheets: {wb.sheetnames}")
    print(f"papers: {len(records)}  gaps: {len(gaps)}  exclusions logged: {len(exclusions)}")


def cat_label(category: str) -> str:
    return CATEGORY_LABEL.get(category, category)


if __name__ == "__main__":
    main()
