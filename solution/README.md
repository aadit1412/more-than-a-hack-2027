# More than a Hack 2027, Phase 1: ball possession

Per-frame possession labels (H / A / D) for BAR-ATH from the provided tracking and Smart Tagging.
Fork of the organisers' starter repository; the solution is in `solution/`.

## Reproduce
1. Unpack the match data into `data/` (or set `MTAH_DATA_DIR`). The data is not included and must not be committed.
2. `pip install -r requirements.txt -r requirements-starter.txt -r solution/requirements.txt`
3. `python solution/make_submission.py`

This writes `submission/possession.csv` (199,319 rows). Expected printout: thresholds 1.85 and 1.49,
label shares D 0.511 / H 0.312 / A 0.176, and 245 possessions with a mean of 12.7 s.
Check the file with `python evaluation/validate_submission.py submission/possession.csv`.

The method and its limitations are in the technical document (`el clasico.md`).
