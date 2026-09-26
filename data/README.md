# Data directory

No data files are committed to this repository. The case-study dataset is published
open access and is fetched by a script, so a fresh clone can obtain it with a
single command and no credentials.

```bash
python scripts/download_data.py     # download, checksum-verify and extract
python scripts/preprocess.py        # validate, clean, feature-engineer and split
```

| Directory | Contents | Tracked? |
| --- | --- | --- |
| `raw/` | the downloaded archive and `download_manifest.json` | no (regenerated) |
| `interim/` | the extracted archive as delivered | no (regenerated) |
| `processed/` | the featured frames and the three chronological splits | no (regenerated) |

Each directory keeps a `.gitkeep` so the structure exists in a fresh clone.

## Source

> Lin, Z., Zhou, Q., Wang, Z., Wang, C., Bookhart, D. B., & Leung-Shea, M. (2024).
> *A high-resolution three-year dataset supporting rooftop photovoltaics (PV)
> generation analytics.* Dryad. <https://doi.org/10.5061/dryad.m37pvmd99>

* Distributed as Zenodo record [10909062](https://zenodo.org/records/10909062)
* Licence: **CC0 1.0** (public-domain dedication — no registration, no API key, no
  click-through, redistributable)
* Archive: 126 files, 296 MB compressed, 1.03 GB extracted
* Selected station: **LSK North** (55.0 kW rated, 15-minute, 2021-06-01 to
  2023-12-31)

`scripts/download_data.py` records the archive URL, byte count and **SHA-256
checksum** in `raw/download_manifest.json` and fails on a mismatch, so a corrupted
or substituted download cannot pass silently.

Selection rationale, the archive layout, the measured quality report and the known
data issues are documented in `docs/dataset.md`.
