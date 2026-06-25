"""Shared test fixtures.

Determinism rule for the project: every stochastic function accepts an injected
`random.Random`, so tests seed it explicitly rather than relying on global state.
"""

import random

import pytest


def pytest_addoption(parser):
    """Add --tier so the integration tests can run against a real architecture
    tier instead of the default tiny CPU-fast smoke config. (pytest requires
    addoption to live in the test-root conftest; the fixtures that consume it
    are in tests/integration/conftest.py.)"""
    parser.addoption(
        "--tier",
        action="store",
        default=None,
        choices=("low", "medium", "high"),
        help="Architecture tier for integration tests (default: tiny smoke config).",
    )


@pytest.fixture
def rng() -> random.Random:
    """A seeded RNG for deterministic tests of stochastic code."""
    return random.Random(1234)
