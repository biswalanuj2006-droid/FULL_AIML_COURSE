"""Ensure a seeded database exists before tests run.

CI starts from an empty checkout - platform.db is gitignored - so the
session fixture seeds (idempotently) before the first test touches the API.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def seed_database():
    from app.seed import run_seed

    result = run_seed()  # no-op when data already exists
    assert result.get("courses", 0) >= 14 or not result.get("seeded"), result
