"""Test tiers (owner decision 2026-11-13; see AGENTS.md).

Tests marked `slow` cover historical pipelines and five-teacher collection end to end. They take about
518 s of a ~912 s suite. The default run skips them and reports them as skipped; `--run-slow` (or
`python scripts/verify_local.py run --with-slow`) runs the full tier.
"""
import pytest


def pytest_addoption(parser):
    parser.addoption("--run-slow", action="store_true", default=False,
                     help="also run tests marked slow (full tier)")


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: long end-to-end test; skipped unless --run-slow is given")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-slow"):
        return
    skip = pytest.mark.skip(reason="slow tier: run with --run-slow")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)
