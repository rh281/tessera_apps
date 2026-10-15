"""The bundled demo AOIs, loaded from ``tessera_apps/data`` with :mod:`importlib.resources`.

Each demo is a GeoJSON FeatureCollection in EPSG:4326. The feature whose ``role`` property is
``"aoi"`` is the AOI (feature 0 in every bundled file); ``caldor.geojson`` also carries the Hansen
2021 loss outline as a second feature with role ``"hansen_2021_loss_outline"``.
"""

from __future__ import annotations

import json
from importlib.resources import files
from typing import Any

from shapely.geometry import shape
from shapely.geometry.base import BaseGeometry

#: Names of the bundled demos, one per app with a demo AOI.
DEMO_NAMES = ("caldor", "rondonia", "everglades")


def _check_name(name: str) -> str:
    key = name.strip().lower()
    if key not in DEMO_NAMES:
        raise ValueError(f"Unknown demo {name!r}; the bundled demos are {', '.join(DEMO_NAMES)}.")
    return key


def demo_geojson(name: str) -> dict[str, Any]:
    """The parsed FeatureCollection of a bundled demo, for example ``demo_geojson("caldor")``.

    Raises:
        ValueError: ``name`` is not one of :data:`DEMO_NAMES`.
    """
    key = _check_name(name)
    text = files("tessera_apps.data").joinpath(f"{key}.geojson").read_text(encoding="utf-8")
    return json.loads(text)


def demo_feature(name: str, role: str) -> BaseGeometry:
    """The geometry of the demo feature with the given ``role`` property.

    Raises:
        ValueError: The demo is unknown or has no feature with that role.
    """
    collection = demo_geojson(name)
    for feature in collection["features"]:
        if (feature.get("properties") or {}).get("role") == role:
            return shape(feature["geometry"])
    roles = [(f.get("properties") or {}).get("role") for f in collection["features"]]
    raise ValueError(f"Demo {name!r} has no feature with role {role!r}; roles present: {roles}.")


def demo_properties(name: str) -> dict[str, Any]:
    """Properties of the demo's AOI feature (name, source, ``year_a``, ``year_b`` and so on)."""
    collection = demo_geojson(name)
    for feature in collection["features"]:
        props = feature.get("properties") or {}
        if props.get("role") == "aoi":
            return dict(props)
    return dict(collection["features"][0].get("properties") or {})


def demo_aoi(name: str) -> BaseGeometry:
    """The AOI polygon of a bundled demo in lon/lat, for example ``demo_aoi("rondonia")``.

    The AOI is the feature with role ``"aoi"``, or the first feature when no feature has a role.

    Raises:
        ValueError: ``name`` is not one of :data:`DEMO_NAMES`.
    """
    collection = demo_geojson(name)
    for feature in collection["features"]:
        if (feature.get("properties") or {}).get("role") == "aoi":
            return shape(feature["geometry"])
    return shape(collection["features"][0]["geometry"])
