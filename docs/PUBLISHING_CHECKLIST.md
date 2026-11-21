# LendScope: pre-publication checklist

This checklist reflects the tracked source exported on 19 November 2026. It is a review aid, **not** a certificate that the full Git history or laptop is free of secrets.

## Blockers: resolve before making the GitHub repository public

- [ ] **Remove the embedded development database password.** The exported `docker-compose.yml` explicitly sets `POSTGRES_PASSWORD`. A safer proposed replacement is included in the publication pack under `proposed_config/docker-compose.yml`. Review its differences and apply only after ensuring the **existing database volume** and app credentials will remain usable. If this password is used outside the local demo, rotate it everywhere it has been reused.
- [ ] **Align `.env.example` with Compose defaults.** Use the companion proposed `.env.example`, and put the actual private password only in `.env` (already Git-ignored). Do not commit `.env` or output credential values when sharing terminal logs.
- [ ] **Check all tracked file contents and history.** Earlier filename-only scanning was not a complete historical content scan. Review unusual URL/token/password matches; run a dedicated secret scanner against all reachable commits before publication.
- [ ] **Review `.streamlit/config.toml` locally.** It was untracked and therefore absent from the source ZIP. Do not publish credentials, cookies or `.streamlit/secrets.toml`. Theme-only settings can be committed after inspection.
- [ ] **Check data use rights.** Confirm the LendingClub dataset's original distribution source, licence, attribution requirements and whether a public download link is appropriate. Do not publish the bulk CSV without permission.
- [ ] **Decide how to handle the password exposed in earlier Git commits.** Removing it from the latest file does not remove it from Git history. For a genuinely sensitive credential, rotate it and consider history remediation before publishing. A throwaway dev-only password that is not reused may only require removal and a clear note.

## Reproducibility

- [ ] Apply and review the secure Compose and `.env.example` examples, then verify `docker compose config --quiet` in the project directory (do not paste full `docker compose config` output; it can contain secrets).
- [ ] On a **new/empty** PostgreSQL volume, confirm `.env` → Compose → `src/ingestion/database.py` → dbt `profiles.yml` are consistent. **Do not remove or reset the user's existing Docker volume** as part of this check.
- [ ] Verify `python -m pip install -r requirements.txt` on a fresh Python 3.11 virtual environment; `pip check` in the existing environment does not establish fresh-install reproducibility.
- [ ] Run `python -m pytest -q` and report the actual test outcome. The prior local result was 51 passed, 1 warning; pandas warns on a deliberately malformed date.
- [ ] Run `dbt debug --project-dir .\dbt`, `dbt run --project-dir .\dbt`, then `dbt test --project-dir .\dbt` against an appropriately populated database. Previous verification was 4 models built and 8 tests passed.
- [ ] Confirm the full six-page Streamlit dashboard loads, including data quality and risk report downloads. The source export did not contain a browser regression-test recording.
- [ ] Preserve the frozen 2015 evaluation. Do not rerun the holdout to retune hyperparameters or represent repeated attempts as a new independent test.

## Repository presentation

- [ ] Copy `README.md`, the `docs/` architecture/checklist files and the sample dbt profile into the repository after review.
- [ ] The author-supplied **Risk Modelling** screenshots are included. Capture a genuine **Overview** and optionally **Data quality** screenshot; redact local information and add image links to `README.md`.
- [ ] Clean or archive untracked historical dashboard revisions and `INSTALL_V*.md` documents outside the repository, without deleting the user's backups.
- [ ] Add a short attribution/licensing section after verifying data and source-licence requirements. Do not invent a software licence; decide whether to add a `LICENSE` file.
- [ ] Check `git status --short`, `.gitignore`, and `git ls-files` before staging. **Do not run `git add .`** with old backups present.
- [ ] Make a reviewed documentation/config commit. Connect a new GitHub remote only after the security review; the exported repository had no remote configured.

## Known design limitations to keep visible

The model uses resolved historical 36-month loans; this introduces selection bias. The split is chronological but not a full point-in-time simulation. ROC-AUC ~0.6565 is moderate, the selected model still underpredicted the 2015 charge-off rate by ~1.84 percentage points, and the work does not establish production performance, fairness, compliance or statistical significance.
