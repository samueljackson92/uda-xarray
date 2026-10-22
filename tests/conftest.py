"""Shared pytest configuration.

Tests with the ``live`` marker read real data, so they need a UDA server. They are skipped
when the server cannot be reached (for example on CI). Run them from a machine with UDA access.
"""

import os
import socket

import pytest

UDA_HOST = os.environ.get("UDA_HOST", "uda2.mast.l")
UDA_PORT = int(os.environ.get("UDA_PORT", "56565"))


def _uda_reachable() -> bool:
    try:
        with socket.create_connection((UDA_HOST, UDA_PORT), timeout=2):
            return True
    except OSError:
        return False


def pytest_configure(config):
    config.addinivalue_line("markers", "live: test needs access to a real UDA server")


def pytest_collection_modifyitems(config, items):
    live_items = [item for item in items if "live" in item.keywords]
    if not live_items or _uda_reachable():
        return
    skip = pytest.mark.skip(reason=f"UDA server {UDA_HOST}:{UDA_PORT} is not reachable")
    for item in live_items:
        item.add_marker(skip)
