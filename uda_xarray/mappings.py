from __future__ import annotations

import importlib.resources
import json
from typing import Iterator

from pydantic import BaseModel


class SignalRange(BaseModel):
    """A single shot-range entry mapping a canonical signal name to the name
    used in the data store for that range of shots."""

    shot_min: int
    shot_max: int
    name: str

    def contains(self, shot: int) -> bool:
        """Return True if *shot* falls within this range (inclusive)."""
        return self.shot_min <= shot <= self.shot_max


class SignalMappings(BaseModel):
    """Complete signal-alias mapping table loaded from mast.json.

    The mapping is a dict whose keys are the canonical (most recent) signal
    names and whose values are lists of :class:`SignalRange` entries ordered
    from newest to oldest.
    """

    mappings: dict[str, list[SignalRange]]

    @classmethod
    def from_file(cls, path: str | None = None) -> "SignalMappings":
        """Load mappings from *path*.

        When *path* is ``None`` all bundled ``*.json`` mapping files are
        loaded and merged into a single :class:`SignalMappings` instance.
        """
        if path is None:
            merged: dict[str, list] = {}
            pkg = importlib.resources.files("uda_xarray")
            for resource in pkg.iterdir():
                if resource.name.endswith(".json"):
                    with importlib.resources.as_file(resource) as p:
                        for k, v in json.loads(p.read_text(encoding="utf-8")).items():
                            if k in merged:
                                merged[k].extend(v)
                            else:
                                merged[k] = list(v)
            raw = merged
        else:
            with open(path, encoding="utf-8") as f:
                raw = json.load(f)

        return cls(mappings={k: [SignalRange(**r) for r in v] for k, v in raw.items()})

    def resolve(self, signal: str, shot: int) -> str | None:
        """Return the stored name for *signal* at the given *shot* number.

        Returns *None* if *signal* is not in the mapping or no range covers
        the shot.
        """
        ranges = self.mappings.get(signal)
        if ranges is None:
            return None
        for r in ranges:
            if r.contains(shot):
                return r.name
        return None

    def __iter__(self) -> Iterator[tuple[str, list[SignalRange]]]:
        return iter(self.mappings.items())

    def __len__(self) -> int:
        return len(self.mappings)
