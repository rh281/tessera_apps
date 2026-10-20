"""The bundled demo AOIs and seed sets, loaded from ``tessera_apps/data`` with :mod:`importlib.resources`.

Each demo is a GeoJSON FeatureCollection in EPSG:4326. The feature whose ``role`` property is
``"aoi"`` is the AOI (feature 0 in every bundled file); ``caldor.geojson`` also carries the Hansen
2021 loss outline as a second feature with role ``"hansen_2021_loss_outline"``.

The restore app's demos (:data:`RESTORE_DEMOS`) are ``itapebi.geojson`` (a eucalyptus planting on
pasture in Bahia, UTM 24S) and ``jaru.geojson`` (natural regrowth on abandoned pasture in Rondonia,
UTM 20S). Each holds three features: role ``"aoi"`` (the site, with properties ``planted`` 2019 and
``synthetic`` true: it was located from public layers, it is not a documented project), role
``"target"`` (a mature forest stand) and role ``"negative"`` (a stand that stayed pasture, also with
``planted`` 2019, the no-divergence example). :func:`demo_roles` lists a demo's roles and
:func:`demo_feature_properties` gives the properties of one of them.

The wetland app's default seeds are bundled next to their demo AOI (:data:`DEMO_SEEDS`):
``everglades_seeds.geojson`` holds 12 square mangrove stand polygons of about 100 m, every feature
with role ``"seed"`` (``scripts/make_everglades_seeds.py`` derives them from ESA WorldCover 2020).
"""

from __future__ import annotations

import json
from importlib.resources import files
from pathlib import Path
from typing import Any

from shapely.geometry import shape
from shapely.geometry.base import BaseGeometry

#: Names of the bundled demos: one AOI each for wildfire (caldor), deforest (rondonia) and wetland
#: (everglades), and the two restore sites.
DEMO_NAMES = ("caldor", "rondonia", "everglades", "itapebi", "jaru")
#: The bundled demos of the restore app, each with ``aoi``, ``target`` and ``negative`` features.
RESTORE_DEMOS = ("itapebi", "jaru")
#: Bundled seed sets, by the demo AOI they belong to (the wetland app's default ``--seeds``).
DEMO_SEEDS = {"everglades": "everglades_seeds"}
#: Names of the bundled seed sets.
SEED_NAMES = tuple(DEMO_SEEDS.values())


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


def demo_roles(name: str) -> list[str | None]:
    """The ``role`` property of every feature of a bundled demo, in file order (None for a feature without one).

    Raises:
        ValueError: The demo is unknown.
    """
    return [(f.get("properties") or {}).get("role") for f in demo_geojson(name)["features"]]


def demo_feature_properties(name: str, role: str) -> dict[str, Any]:
    """Properties of the demo feature with the given ``role`` property, for example its ``planted`` year.

    Raises:
        ValueError: The demo is unknown or has no feature with that role.
    """
    for feature in demo_geojson(name)["features"]:
        props = feature.get("properties") or {}
        if props.get("role") == role:
            return dict(props)
    raise ValueError(f"Demo {name!r} has no feature with role {role!r}; roles present: {demo_roles(name)}.")


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


def _data_name(name: str) -> str:
    key = name.strip().lower()
    if key not in DEMO_NAMES and key not in SEED_NAMES:
        known = ", ".join((*DEMO_NAMES, *SEED_NAMES))
        raise ValueError(f"Unknown bundled data {name!r}; the bundled files are {known}.")
    return key


def demo_path(name: str) -> Path:
    """Path of a bundled GeoJSON: a demo AOI (``"everglades"``) or a seed set (``"everglades_seeds"``).

    Raises:
        ValueError: ``name`` is neither a demo nor a seed set.
    """
    return Path(str(files("tessera_apps.data").joinpath(f"{_data_name(name)}.geojson")))


def seeds_name(name: str) -> str:
    """The seed set bundled for a demo AOI, or the seed set itself: ``seeds_name("everglades")``.

    Raises:
        ValueError: ``name`` is not a seed set or a demo that has one.
    """
    key = name.strip().lower()
    if key in SEED_NAMES:
        return key
    if key in DEMO_SEEDS:
        return DEMO_SEEDS[key]
    have = ", ".join(f"{k} ({v})" for k, v in DEMO_SEEDS.items())
    raise ValueError(f"No bundled seeds for {name!r}; seeds are bundled for {have}.")


def demo_seeds(name: str) -> dict[str, Any]:
    """The parsed seed FeatureCollection bundled for a demo AOI or named directly.

    Every feature has role ``"seed"``. For example ``demo_seeds("everglades")`` gives the 12
    mangrove stand squares of the Everglades demo.

    Raises:
        ValueError: ``name`` is not a seed set or a demo that has one.
    """
    key = seeds_name(name)
    text = files("tessera_apps.data").joinpath(f"{key}.geojson").read_text(encoding="utf-8")
    return json.loads(text)
