"""Small, synthetic fixtures. No database, external data, or trained models required."""

import sys
from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture
def raw_feature_chunk():
    """Matches transform_chunk's SQL aliases, not raw LendingClub column names."""
    return pd.DataFrame({
        "loan_id": [1, 2],
        "issue_date": ["2014-01-01", "2014-05-01"],
        "earliest_cr_line": ["Jan-2010", "Apr-2014"],
        "revol_util": ["85.0", None],
        "inq_last_6mths": ["2", ""],
        "delinq_2yrs": ["0", "0"],
        "pub_rec": ["1", None],
        "open_acc": ["8", "10"],
        "employment_length": [" 10+ years ", None],
        "verification_status": [" Verified ", ""],
        "target": [0, 1],
    })
