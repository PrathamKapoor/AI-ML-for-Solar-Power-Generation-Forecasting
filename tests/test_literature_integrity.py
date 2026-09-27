"""Literature-set integrity: ids, categories, evidence and the workbook.

The literature review is the one part of the repository that is maintained by
hand rather than computed, so it is the part that can quietly go wrong. Each
test here pins an invariant whose violation was found in the stored artefacts:

* a paper id whose prefix disagrees with its category, so a reader mapping
  ``G4`` to "Systematic review" finds it filed under classical ML;
* a gap citing the same paper twice, which inflated the evidence count that a
  reader takes as the strength of the claim;
* a category tally that silently disagreed with the corpus size, because an id
  present in one source and not the other was skipped instead of reported.

The workbook tests need the built ``.xlsx``; they skip when it is absent so the
suite still runs in a fresh clone before the literature pipeline has been run.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools" / "literature"
WORKBOOK = ROOT / "literature" / "literature_review.xlsx"
EVIDENCE = ROOT / "literature" / "verification" / "gap_evidence.json"

pytestmark = pytest.mark.skipif(
    not (TOOLS / "final_papers.json").exists(),
    reason="literature sources are not present")


def load(name: str) -> dict:
    return json.loads((TOOLS / name).read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# Identity
# --------------------------------------------------------------------------- #
def test_the_corpus_holds_exactly_thirty_papers() -> None:
    papers = load("final_papers.json")["papers"]
    assert len(papers) == 30
    assert len({p["id"] for p in papers}) == 30, "paper ids must be unique"


def test_a_paper_id_prefix_matches_its_category() -> None:
    """``G4`` filed under category A is a trap for anyone reading the ids.

    A reader maps the id prefix to the taxonomy from the category column and the
    paper is not where the id says it is. It also collided with the ``G1..G6``
    section labels in ``docs/research_gaps.md``.
    """
    for paper in load("final_papers.json")["papers"]:
        assert paper["id"].startswith(paper["category"]), (
            f"id {paper['id']!r} is filed under category {paper['category']!r}")


def test_the_paper_order_follows_the_taxonomy() -> None:
    papers = load("final_papers.json")["papers"]
    keys = [(p["category"], int(p["id"][1:])) for p in papers]
    assert keys == sorted(keys), "the workbook numbers rows in list order"


def test_no_two_papers_share_a_doi() -> None:
    papers = load("final_papers.json")["papers"]
    dois = [p["doi"] for p in papers if p.get("doi")]
    assert len(dois) == len(set(dois))


# --------------------------------------------------------------------------- #
# Evidence
# --------------------------------------------------------------------------- #
def test_every_cited_evidence_paper_exists_and_is_cited_once() -> None:
    """A repeated citation inflates the count a reader reads as strength."""
    known = {p["id"] for p in load("final_papers.json")["papers"]}
    for gap in load("research_gaps.json")["gaps"]:
        cited = [p.strip() for p in gap["evidence_papers"].split(",") if p.strip()]
        assert cited, f"{gap['gap_id']} cites no evidence"
        dangling = [p for p in cited if p not in known]
        assert not dangling, f"{gap['gap_id']} cites unknown papers {dangling}"
        dupes = {p for p in cited if cited.count(p) > 1}
        assert not dupes, f"{gap['gap_id']} cites {sorted(dupes)} more than once"


def test_the_stored_evidence_summary_agrees_with_itself() -> None:
    """``category_counts`` must sum to ``corpus_size``.

    Ids present in one source and not the other used to be skipped silently, so
    the tally could report 29 against a corpus of 30 without any error.
    """
    if not EVIDENCE.exists():
        pytest.skip("gap evidence has not been built")
    payload = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    assert sum(payload["category_counts"].values()) == payload["corpus_size"]
    assert not payload.get("dangling_evidence_references")
    assert not payload.get("repeated_evidence_references")
    for gap in payload["gaps"]:
        assert gap["n_cited"] == len(set(gap["cited_papers"])), gap["gap_id"]


def test_every_retained_gap_states_a_reason_for_both_outcomes() -> None:
    """Triage is recorded rather than applied silently."""
    triage = load("research_gaps.json")["triage"]
    assert triage, "the triage decision must be recorded"
    for gap in load("research_gaps.json")["gaps"]:
        assert gap.get("retained") is not None, gap["gap_id"]
        assert gap.get("triage_reason"), f"{gap['gap_id']} has no triage reason"


# --------------------------------------------------------------------------- #
# Workbook
# --------------------------------------------------------------------------- #
REVIEW_COLUMNS = ["Sr No", "Name", "Author(s)", "Publishing Date",
                  "Published By / Organization", "Title",
                  "Abstract (review summary)", "Conclusion (review summary)",
                  "Keywords Related", "Paper Link"]


@pytest.mark.skipif(not WORKBOOK.exists(), reason="workbook has not been built")
def test_the_review_sheet_has_the_required_columns_and_no_blanks() -> None:
    pd = pytest.importorskip("pandas")
    frame = pd.read_excel(WORKBOOK, sheet_name="Literature Review")
    assert list(frame.columns) == REVIEW_COLUMNS
    assert len(frame) == 30
    for column in REVIEW_COLUMNS:
        assert frame[column].notna().all(), f"{column} has blanks"
        assert not frame[column].astype(str).str.strip().eq("").any(), column
    assert not frame["Paper Link"].duplicated().any()
    assert not frame["Title"].duplicated().any()


@pytest.mark.skipif(not WORKBOOK.exists(), reason="workbook has not been built")
def test_the_analysis_sheet_carries_the_twenty_nine_columns() -> None:
    pd = pytest.importorskip("pandas")
    frame = pd.read_excel(WORKBOOK, sheet_name="Literature Analysis")
    assert len(frame.columns) == 29
    assert len(frame) == 30
    for column in frame.columns:
        assert frame[column].notna().all(), f"{column} has blanks"


@pytest.mark.skipif(not WORKBOOK.exists(), reason="workbook has not been built")
def test_the_gap_sheet_records_the_triage_decision() -> None:
    """Retained and dropped candidates share the sheet, with the reason beside them."""
    pd = pytest.importorskip("pandas")
    frame = pd.read_excel(WORKBOOK, sheet_name="Research Gaps")
    for column in ("Gap ID", "Status", "Triage Role", "Triage Reason"):
        assert column in frame.columns, column
    assert frame["Triage Reason"].notna().all()
    statuses = set(frame["Status"])
    assert statuses <= {"retained", "not retained"}, statuses
    assert "retained" in statuses


@pytest.mark.skipif(not WORKBOOK.exists(), reason="workbook has not been built")
def test_every_figure_referenced_by_the_paper_exists() -> None:
    """The paper must not cite a figure that was never produced."""
    readme = (ROOT / "paper" / "figures" / "README.md")
    if not readme.exists():
        pytest.skip("figure index has not been generated")
    text = readme.read_text(encoding="utf-8")
    listed = set(re.findall(r"\| `([^`]+\.png)` \|", text))
    on_disk = {p.name for p in (ROOT / "results" / "figures").glob("*.png")}
    assert listed <= on_disk, f"index names absent figures: {sorted(listed - on_disk)}"
