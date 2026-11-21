"""Checks for reproducible model configuration and safe environment handling."""

import pytest

from src.ingestion import database
from src.modeling.feature_ablation import REDUCED_FEATURES, REMOVED_FEATURES, make_reduced_model
from src.modeling.train_baseline import FEATURES as BASE_FEATURES
from src.modeling.train_expanded_model import FEATURES as EXPANDED_FEATURES


def test_reduced_feature_set_is_frozen_and_excludes_post_origination_fields():
    assert len(BASE_FEATURES) == 6
    assert len(EXPANDED_FEATURES) == 14
    assert len(REDUCED_FEATURES) == 12
    assert set(REDUCED_FEATURES) == set(EXPANDED_FEATURES) - set(REMOVED_FEATURES)
    assert set(BASE_FEATURES).issubset(REDUCED_FEATURES)
    assert not ({"loan_status", "total_pymnt", "recoveries", "last_pymnt_d"} & set(REDUCED_FEATURES))


def test_reduced_pipeline_uses_exactly_the_selected_features():
    model = make_reduced_model()  # Constructs pipeline; does not fit or train.
    preprocessing = model.named_steps["preprocessor"]
    selected = [f for _, _, cols in preprocessing.transformers for f in cols]
    assert set(selected) == set(REDUCED_FEATURES)
    assert len(selected) == len(REDUCED_FEATURES)
    assert model.named_steps["classifier"].max_iter == 160
    assert model.named_steps["classifier"].random_state == 42


@pytest.fixture
def fake_db_environment(monkeypatch):
    settings = {
        "POSTGRES_HOST": "127.0.0.1", "POSTGRES_PORT": "5433",
        "POSTGRES_DB": "synthetic_db", "POSTGRES_USER": "synthetic_user",
        "POSTGRES_PASSWORD": "synthetic_password",
    }
    for key, value in settings.items():
        monkeypatch.setenv(key, value)
    return settings


def test_db_url_uses_explicit_environment_configuration(fake_db_environment):
    url = database.get_database_url()
    assert url.startswith("postgresql+psycopg://synthetic_user:")
    assert "@127.0.0.1:5433/synthetic_db" in url


def test_db_configuration_fails_when_required_secret_is_missing(fake_db_environment, monkeypatch):
    monkeypatch.delenv("POSTGRES_PASSWORD")
    with pytest.raises(RuntimeError, match="POSTGRES_PASSWORD"):
        database.get_database_url()
