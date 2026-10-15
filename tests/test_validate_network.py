"""Network tier for validate.py: each reference loader against its live source on a demo AOI.

Run with ``pytest -m network tests/test_validate_network.py``. The Hansen and Natural Earth tests
reach hosts the build sandbox can reach (storage.googleapis.com, naturalearth.s3.amazonaws.com).
NIFC, EFFIS, GMW (Zenodo) and USFS FACTS were blocked from the build sandbox, so those tests are
expected to fail there with ReferenceUnavailableError naming the host; they pass where the hosts
are reachable. The zone plans come from the live TESSERA store (metadata only, no chunk reads),
opened with AWS credentials cleared so the anonymous read is what gets tested.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
import shapely
from pyproj import Geod
from shapely.geometry import box

from tessera_apps import validate as v
from tessera_apps.demo import demo_aoi, demo_feature
from tessera_apps.io import AOIPlan, TesseraReader, plan_aoi

pytestmark = pytest.mark.network


@pytest.fixture(scope="module", autouse=True)
def live_cache(tmp_path_factory) -> Iterator[None]:
    """A fresh reference cache for every test, whatever the selection or order, so each source is
    fetched for real and never answered from the user's cache; and no AWS credentials.
    """
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("TESSERA_APPS_CACHE", str(tmp_path_factory.mktemp("cache")))
        for key in [k for k in os.environ if k.startswith("AWS_")]:
            mp.delenv(key)
        yield


@pytest.fixture(scope="module")
def reader(live_cache) -> TesseraReader:
    return TesseraReader.open()


def demo_plan(reader: TesseraReader, name: str) -> AOIPlan:
    return plan_aoi(reader, demo_aoi(name))


def geodesic_ha(geoms) -> float:
    union = shapely.union_all(list(geoms))
    return abs(Geod(ellps="WGS84").geometry_area_perimeter(union)[0]) / 1e4


def test_hansen_rondonia_loss_forest_valid(reader):
    """Survey S14: about 22 percent loss in 2020 to 2022 and about 61 percent forest in 2019."""
    aoi = demo_plan(reader, "rondonia")
    assert aoi.zone_names == ["20S"]
    plan = aoi.zones[0]
    assert v.hansen_tiles(v.plan_lonlat_bounds(plan)) == ["00N_070W"]
    loss = v.hansen_loss(plan, 2019, 2022)
    forest = v.hansen_forest(plan, 2019)
    valid = v.hansen_valid(plan)
    n = plan.n_pixels
    loss_frac, forest_frac, valid_frac = loss.sum() / n, forest.sum() / n, valid.sum() / n
    assert 0.18 < loss_frac < 0.27, loss_frac
    assert 0.5 < forest_frac < 0.7, forest_frac
    assert valid_frac > 0.95, valid_frac
    assert not (loss & ~plan.mask).any()
    # Loss after 2019 can only fall on pixels that were not lost by 2019.
    lossyear = v.hansen_layer(plan, "lossyear")
    assert not (loss & (lossyear >= 1) & (lossyear <= 19)).any()


def test_natural_earth_everglades_coastline():
    bbox = demo_aoi("everglades").bounds
    lines = v.natural_earth_coastline(bbox)
    assert len(lines) >= 2
    area = box(*bbox).buffer(1e-9)
    assert all(area.contains(g) for g in lines)
    km = sum(Geod(ellps="WGS84").geometry_length(g) for g in lines) / 1000
    assert 50 < km < 400, km


def test_nifc_caldor_perimeter():
    """Caldor burned 221,835 acres (89,774 ha); the perimeter crosses 120 W into zone 11N (survey S9)."""
    outline = demo_feature("caldor", "hansen_2021_loss_outline")
    geoms = v.nifc_perimeters(demo_aoi("caldor").bounds, year=2021, name="Caldor")
    assert geoms, "no Caldor 2021 perimeter returned"
    ha = geodesic_ha(geoms)
    assert 80_000 < ha < 100_000, ha
    union = shapely.union_all(geoms)
    assert union.intersects(outline)
    assert union.bounds[2] > -120.0


def test_effis_evia_2021():
    """The August 2021 north Evia fire burned about 51,000 ha."""
    geoms = v.effis_burnt_areas((22.9, 38.7, 23.6, 39.1), year=2021)
    assert geoms
    assert geodesic_ha(geoms) > 30_000


def test_gmw_v3_everglades_2018(reader):
    plan = demo_plan(reader, "everglades").zones[0]
    extent = v.gmw_extent(plan, 2018, version="v3")
    frac = extent.sum() / plan.n_pixels
    assert 0.1 < frac < 0.8, frac


def test_usfs_reforestation_king_fire():
    """Plantings in the 2014 King Fire scar, Eldorado NF, completed 2019 to 2020 (survey S16)."""
    frame = v.usfs_reforestation((-120.75, 38.75, -120.35, 39.05), completed_from=2019, completed_to=2020)
    assert len(frame) > 0
    years = frame["DATE_COMPLETED"].dt.year
    assert years.between(2019, 2020).all()
    assert np.isfinite(frame.total_bounds).all()


def test_every_network_test_uses_the_fresh_cache(request):
    """Run alone, a network test must still read and write a throwaway cache, not ~/.cache."""
    assert "live_cache" in request.fixturenames
    assert v.cache_dir() != Path.home() / ".cache" / "tessera_apps"
    # Downloads and cache entries go through export.atomic_path, whose scratch names end in ".tmp".
    assert not [p for p in v.cache_dir().rglob("*") if p.name.endswith(".tmp") or ".part" in p.name]
