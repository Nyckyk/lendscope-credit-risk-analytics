# LendScope automated test suite

This package adds **51 pytest checks** to the existing credit-risk-data-platform repository. It does **not** include or replace source modules, dashboard code, model files, training data, SQL, or reports.

## Install on Windows

1. Save `lendscope_pytest_suite.zip` (usually to `Downloads`).
2. In PowerShell, run the commands below, replacing `$Zip` with its actual location if necessary:

```powershell
$Project = 'C:\Users\nicho\AI projects\credit-risk-data-platform'
$Zip = Join-Path $HOME 'Downloads\lendscope_pytest_suite.zip'

if (-not (Test-Path -LiteralPath $Zip)) { throw "ZIP not found: $Zip" }
if (Test-Path -LiteralPath (Join-Path $Project 'tests')) { throw 'Tests folder already exists. Review before merging.' }
if (Test-Path -LiteralPath (Join-Path $Project 'pytest.ini')) { throw 'pytest.ini already exists. Review before replacing.' }

Expand-Archive -LiteralPath $Zip -DestinationPath $Project
Set-Location -LiteralPath $Project
& .\.venv\Scripts\Activate.ps1
python -m pytest -q
```

The archive contains `TESTING.md`, `pytest.ini` and `tests/` in its root. `pytest.ini` fixes the test discovery path, so `python -m pytest -q` runs only the intended tests.

## Coverage

- **Feature engineering:** numeric fields, missing categories, credit-history months, preservation of input frames, and rejection of invalid values or dates.
- **Input data and cohort contracts:** LendingClub source columns, duplicate/missing loan IDs, binary targets and original/expanded cohort matching.
- **Model evaluation:** ROC-AUC, average precision, Brier score, decile assignment, ranking ties, summary bias, and probability-range validation.
- **Final holdout loader:** 2015 issue-year checks, loan-ID alignment, duplicate IDs and mismatched targets, all using three-row test CSVs.
- **Risk Modelling page reports:** validates the app's real `load_risk_reports()` and `load_reduced_importance()` functions using synthetic reports. Includes failure tests for wrong cohort counts, non-finite values, inconsistent decile counts, incorrect predicted means, SHA-256 mismatches and invalid shuffle deviations.
- **Frozen model configuration:** verifies the selected feature list and pipeline configuration without training a model.
- **Database configuration:** validates presence of the environment variables using invented credentials, without connecting to PostgreSQL.

### Scope and caveats

- These are deterministic **unit/contract tests**, not full dashboard/browser integration tests or proof of the statistical validity of the risk model.
- They do not use PostgreSQL, Docker, internet access, real loan CSVs, trained `.joblib` files, or the saved 2015 holdout. `load_and_verify_test_data()` is tested using local temporary CSVs only.
- The dashboard report functions are parsed directly from `dashboard/app.py` and executed independently (not rendered through Streamlit). This avoids Streamlit UI and database startup side effects.
- Keep running `dbt test --project-dir .\dbt` separately for the eight live data tests. A successful pytest run does not replace dbt testing.
- Tested against the source ZIP provided in this conversation. In the build environment, `51 passed, 1 warning`; the warning is from pandas date-format inference on a deliberately malformed test date. Verify results on your own Python 3.11 environment.
- Relevant dependencies already used by the project: pytest, pandas, numpy, scipy, scikit-learn, joblib, SQLAlchemy and python-dotenv. Install missing packages in your existing virtual environment if your local pytest run identifies any.

## Version control

After the suite passes locally:

```powershell
git status --short
git add tests pytest.ini TESTING.md
git diff --cached --check
git diff --cached --stat
git commit -m "Add synthetic-data tests for credit risk pipeline"
```

Do not run `git add .`: old dashboard backups and local configuration files are still untracked in this repository.
