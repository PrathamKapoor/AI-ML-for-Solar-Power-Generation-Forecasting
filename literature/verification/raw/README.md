# Raw API payloads (not committed)

This directory is where `tools/literature/verify_papers.py` writes the
unmodified JSON payloads returned by Crossref, OpenAlex and Semantic Scholar
during metadata verification.

The files are intentionally **not committed**: there are ~90 of them and they
are fully reproducible by rerunning

`ash
python tools/literature/resolve_seeds.py
python tools/literature/verify_papers.py
python tools/literature/fetch_abstracts.py
`

The consolidated, committed audit trail lives one level up in
`verified_papers.json`, which records for every paper which sources returned a
record, which source supplied the abstract, and the Crossref/OpenAlex title
agreement score.
