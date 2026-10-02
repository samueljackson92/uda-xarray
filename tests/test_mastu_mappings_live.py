"""Live tests for the MAST-U signal mappings in ``mastu.json``.

These tests read real data, so they need a UDA server. They are skipped when the server
cannot be reached (for example on CI). Run them from a machine with UDA access.
"""

import os
import socket

import pytest
import xarray as xr

from uda_xarray.mappings import SignalMappings

UDA_HOST = os.environ.get("UDA_HOST", "uda2.mast.l")
UDA_PORT = int(os.environ.get("UDA_PORT", "56565"))

# Renamed signals. The key is the current name. A shot with the current name needs no mapping.
RENAMED = (
    "/ESM/VLOOP/DYNAMIC",
    "/ESM/VLOOP/STATIC",
    "/ESM/DENSITY/NEBAR",
    "/ESM/DENSITY/NG",
    "/ESM/DENSITY/FG",
    "/ESM/POWER/POHM",
    "/XDC/AI/CPU1/ROGEXT_TF",
)
# Shots that have the current name of every signal in RENAMED, and are not in a mapped range.
# 47000 has both the old and the current names. 52002 is after the old names end (shot 51257).
CURRENT_NAME_SHOTS = (47000, 52002)
# Number of shots to read for each signal. Each read takes a few seconds.
SAMPLES_PER_SIGNAL = 5


def _uda_reachable() -> bool:
    try:
        with socket.create_connection((UDA_HOST, UDA_PORT), timeout=2):
            return True
    except OSError:
        return False


pytestmark = pytest.mark.skipif(
    not _uda_reachable(), reason=f"UDA server {UDA_HOST}:{UDA_PORT} is not reachable"
)

MAPPINGS = SignalMappings.from_file()


def _sample_shots(signal: str) -> list[int]:
    """Return shots spread over the mapped ranges, including the first and the last."""
    ranges = sorted(MAPPINGS.mappings[signal], key=lambda r: r.shot_min)
    shots = [ranges[0].shot_min, ranges[-1].shot_max]
    step = max(1, len(ranges) // (SAMPLES_PER_SIGNAL - 1))
    shots += [r.shot_min for r in ranges[::step]]
    return sorted(set(shots))[: SAMPLES_PER_SIGNAL + 1]


def _open(name: str, shot: int) -> xr.Dataset:
    return xr.open_dataset(f"uda://{name}:{shot}", engine="uda")


def _cases() -> list[tuple[str, int]]:
    return [(signal, shot) for signal in RENAMED for shot in _sample_shots(signal)]


@pytest.mark.parametrize(("signal", "shot"), _cases())
def test_mapped_shot_returns_data_under_old_name(signal, shot):
    """The current name is read from the old name in a mapped shot."""
    old_name = MAPPINGS.resolve(signal, shot)
    assert old_name is not None

    dataset = _open(signal.lstrip("/").lower(), shot)

    assert dataset["data"].size > 0
    assert dataset["data"].attrs["uda_name"] == old_name


@pytest.mark.parametrize("shot", CURRENT_NAME_SHOTS)
@pytest.mark.parametrize("signal", RENAMED)
def test_current_name_shot_is_not_remapped(signal, shot):
    """A shot with the current name is not remapped, and still returns data."""
    assert MAPPINGS.resolve(signal, shot) is None

    dataset = _open(signal.lstrip("/").lower(), shot)

    assert dataset["data"].size > 0
    assert dataset["data"].attrs["uda_name"].lstrip("/").upper() == signal.lstrip("/")
