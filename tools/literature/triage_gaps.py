"""Mark which research gaps are retained, with the reason for each decision.

    python tools/literature/triage_gaps.py

An earlier pass listed ten candidate gaps. Four did not survive scrutiny: their
evidence either pointed the other way, or the real gap was narrower and already
covered by a retained theme. Deleting them would remove the record of the
decision, so they are kept in ``research_gaps.json`` and marked ``retained: false``
with the reason. The workbook and the prose then report the retained set, and
``docs/research_gaps.md`` records the four drops with their reasons.

The four retained themes are the ones the evidence supports:

* RG-02 aggregate-only evaluation, with condition-dependence unreported;
* RG-05 horizon sensitivity asserted rather than measured;
* RG-08 the classical arm reported untuned, and cost not reported with accuracy;
* RG-07 feature importance from one method, or from attention weights.

RG-01, RG-03, RG-04, RG-06, RG-09 and RG-10 are *supporting* observations rather
than gaps in their own right: each is something this project does rather than a
gap the literature leaves open, and several are answered by a retained theme. They
are kept and labelled so the workbook shows the full candidate set while the claim
set stays defensible.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GAPS = ROOT / "tools" / "literature" / "research_gaps.json"

# gap_id -> (retained, role, reason)
TRIAGE: dict[str, tuple[bool, str, str]] = {
    "RG-01": (True, "retained",
              "The central gap this project addresses: no shared protocol places the "
              "classical, recurrent and attention families side by side on one "
              "leakage-audited split."),
    "RG-02": (True, "retained",
              "Aggregate-only evaluation dominates, and a ranking change between "
              "conditions is almost never reported."),
    "RG-03": (False, "answered_by_practice",
              "A real reporting weakness, but this project answers it rather than "
              "documenting a gap: every comparison here is formally tested with a "
              "block bootstrap and a Diebold-Mariano procedure."),
    "RG-04": (False, "answered_by_practice",
              "A genuine defect in the literature and one this project is careful "
              "about, with chronological splits and explicit leakage tests. It is a "
              "requirement of the work, not a gap the work leaves open."),
    "RG-05": (True, "retained",
              "Horizon sensitivity is asserted rather than measured, which this "
              "project measures directly and finds refutes the usual claim."),
    "RG-06": (False, "narrower_than_stated",
              "The initial claim was that calibration is rarely reported. The "
              "probabilistic PV literature is methodologically careful and does "
              "report it; the real gap is narrower, that the deterministic majority "
              "reports no interval at all, and it is not this paper's contribution."),
    "RG-07": (True, "retained",
              "Feature importance is reported from one method, or from attention "
              "weights treated as explanations, and a single ranking cannot be "
              "checked against anything."),
    "RG-08": (True, "retained",
              "The classical arm is reported untuned and cost is almost never "
              "reported alongside the accuracy it is meant to justify, so a fair "
              "accuracy-versus-cost frontier cannot be drawn."),
    "RG-09": (False, "answered_by_practice",
              "The observation is correct and unusual, but it is a virtue of this "
              "repository rather than a gap in the literature that the paper sets "
              "out to close."),
    "RG-10": (False, "answered_by_practice",
              "Metric inconsistency is real and this project documents every metric, "
              "but it is a matter of hygiene rather than a research gap."),
    "RG-11": (True, "retained",
              "Retained: the reference's competence is a function of the horizon, so "
              "the choice of persistence reference is a substantive design decision "
              "rather than a formality."),
    "RG-12": (True, "retained",
              "Retained: this is the clearest methodological gap in the reproduction "
              "literature, and it is where the project obtains its most consequential "
              "result, which is a negative one."),
}


def main() -> int:
    if not GAPS.exists():
        print(f"missing {GAPS}", file=sys.stderr)
        return 1
    payload = json.loads(GAPS.read_text(encoding="utf-8"))
    gaps = payload.get("gaps", [])
    if not gaps:
        print("no gaps in the file", file=sys.stderr)
        return 1

    unknown = sorted({g.get("gap_id") for g in gaps} - set(TRIAGE))
    if unknown:
        print(f"undecided gap ids: {unknown}; add them to TRIAGE before writing",
              file=sys.stderr)
        return 1

    for gap in gaps:
        retained, role, reason = TRIAGE[gap["gap_id"]]
        gap["retained"] = retained
        gap["triage_role"] = role
        gap["triage_reason"] = reason

    payload["triage"] = {
        "n_total": len(gaps),
        "n_retained": sum(1 for g in gaps if g["retained"]),
        "policy": ("A gap is retained only where the evidence supports it and the "
                   "project does not itself answer it. Everything else is kept in "
                   "this file, marked not retained, with the reason, so the "
                   "decision is auditable rather than invisible."),
    }
    GAPS.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print(f"gaps total    : {payload['triage']['n_total']}")
    print(f"gaps retained : {payload['triage']['n_retained']}")
    for gap in gaps:
        mark = "retain " if gap["retained"] else "drop   "
        print(f"  {mark} {gap['gap_id']}  {gap['triage_role']}")
    print(f"written       : {GAPS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
