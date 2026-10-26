"""The hosted demo's map click (``demo/trajectory.py``) on a designed in-memory store, offline.

The store is one 10N group of 3 x 3 chunks of 64 x 64 pixels, one chunk per shard, so a KeyLog
names exactly the chunks a click reads. Forest everywhere; a 32 x 32 scar in chunk (1, 1) that is
forest before 2021 and turns to ash and back from 2021; open water (no data in any year) in chunk
(0, 2); and two radar gaps inside the scar, in 2022 and in 2024. The click's numbers are checked
against the wildfire app's own full run on that chunk as its AOI (the window's own rules), and
against its full run on the whole grid (a precomputed run's settings, as the page takes the
precomputed Caldor run's). A second store adds what makes the app warn: an unburned reference that
drifted, a radar-free scar and a stable pair that is not stable.
"""

from __future__ import annotations

import dataclasses
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import rasterio
import zarr
from pyproj import Transformer
from rasterio.transform import rowcol
from shapely.geometry import mapping
from zarr.storage import MemoryStore

from tessera_apps import metrics
from tessera_apps.apps import wildfire
from tessera_apps.io import N_BANDS, YEARS, StoreUnavailableError, TesseraReader

from .conftest import KeyLog, designed_zone, grid_polygon, in_box, restore_corner

ROOT = Path(__file__).resolve().parent.parent
if not (ROOT / "demo" / "trajectory.py").is_file():  # an unpacked sdist has tests/ but not demo/
    pytest.skip("demo/trajectory.py is not in this tree (the sdist does not ship demo/).", allow_module_level=True)

from demo import trajectory  # noqa: E402
from demo.trajectory import ClickError, DemoStore, WindowCache, run_click  # noqa: E402

SNAP = "TESTSNAPSHOT0000000"
CHUNK = 64
SIZE = 3 * CHUNK
CENTRE = (-120.4, 38.8)
LEFT, TOP = restore_corner(CENTRE)
_RNG = np.random.default_rng(81)
FOREST = _RNG.integers(-40, 41, size=N_BANDS).astype(np.float64)
ASH = _RNG.integers(-40, 41, size=N_BANDS).astype(np.float64)
#: ((row0, row1), (col0, col1)) on the zone grid. The scar lies inside chunk (1, 1).
SCAR = ((80, 112), (72, 104))
WATER = ((0, 64), (128, 192))
RADAR_GAP_2022 = ((100, 104), (90, 94))
RADAR_GAP_2024 = ((86, 90), (76, 80))
#: Gaps in single years, in chunk (2, 0).
HOLE_2022 = ((150, 160), (10, 20))
HOLE_2024 = ((170, 180), (10, 20))
#: The scar's share of ash by year: the fire year half seen, then a slow return to forest.
ASH_WEIGHT = {2021: 0.5, 2022: 1.0, 2023: 0.8, 2024: 0.6, 2025: 0.4}
YEARS_DEFAULT = [2020, 2021, 2022, 2023, 2024, 2025]
#: The default years and the stable pair's 2019, which the window's own rules read for the
#: threshold fallback or for the check of the window's unburned reference.
YEARS_WITH_PAIR = [2019, *YEARS_DEFAULT]
LAYERS = set(wildfire.LAYERS)
#: Chunks of the grid, as ((row0, row1), (col0, col1)).
CHUNK_11 = ((64, 128), (64, 128))
CHUNK_22 = ((128, 192), (128, 192))
WHOLE_GRID = ((0, SIZE), (0, SIZE))
#: The second store: the rest of chunk (1, 1) is singed (a little ash from 2021, under half the
#: threshold, so it is in the unburned reference and makes it drift), the scar has no Sentinel-1 in
#: any year, and chunk (2, 2) has ash in 2019 alone, so its stable pair 2019 to 2020 is not stable.
SINGED_WEIGHT = 0.3
UNSTABLE_2019_WEIGHT = 0.3


def _prototypes(year: int) -> np.ndarray:
    w = ASH_WEIGHT.get(year, 0.0)
    return np.stack([FOREST, (1.0 - w) * FOREST + w * ASH])


def _pattern(rows: np.ndarray, cols: np.ndarray, year: int) -> tuple[np.ndarray, ...]:
    classes = in_box(rows, cols, SCAR).astype(np.int64)
    data = ~in_box(rows, cols, WATER)
    if year == 2022:
        data &= ~in_box(rows, cols, HOLE_2022)
    if year == 2024:
        data &= ~in_box(rows, cols, HOLE_2024)
    radar = np.ones(rows.shape, dtype=bool)
    if year == 2022:
        radar &= ~in_box(rows, cols, RADAR_GAP_2022)
    if year == 2024:
        radar &= ~in_box(rows, cols, RADAR_GAP_2024)
    return classes, data, radar


def _warning_prototypes(year: int) -> np.ndarray:
    singed = SINGED_WEIGHT if year >= 2021 else 0.0
    unstable = UNSTABLE_2019_WEIGHT if year == 2019 else 0.0
    return np.concatenate(
        [
            _prototypes(year),
            np.stack([(1.0 - singed) * FOREST + singed * ASH, (1.0 - unstable) * FOREST + unstable * ASH]),
        ]
    )


def _warning_pattern(rows: np.ndarray, cols: np.ndarray, year: int) -> tuple[np.ndarray, ...]:
    classes, data, radar = _pattern(rows, cols, year)
    scar = in_box(rows, cols, SCAR)
    classes = np.where(in_box(rows, cols, CHUNK_11) & ~scar, 2, classes)
    classes = np.where(in_box(rows, cols, CHUNK_22), 3, classes)
    return classes, data, radar & ~scar


def _mem(
    years_complete: list[int] | None = None,
    *,
    pattern: Any = _pattern,
    prototypes: Any = _prototypes,
) -> MemoryStore:
    mem = MemoryStore()
    root = zarr.open_group(mem, mode="w")
    root.attrs.update({"checkpoint_id": "tessera_v1_1_aws_encoder.pt", "geoemb:dimensions": N_BANDS})
    designed_zone(
        root,
        "10N",
        left=LEFT,
        top=TOP,
        height=SIZE,
        width=SIZE,
        pattern=pattern,
        prototypes=prototypes,
        chunk=CHUNK,
        years_complete=years_complete,
    )
    return mem


@pytest.fixture(scope="module")
def scar_mem() -> MemoryStore:
    return _mem()


@pytest.fixture
def demo(scar_mem) -> tuple[DemoStore, KeyLog]:
    log = KeyLog(scar_mem)
    reader = TesseraReader.from_store(log, snapshot_id=SNAP)
    return DemoStore.from_reader(reader), log


_TO_LONLAT = Transformer.from_crs(32610, 4326, always_xy=True)


def lonlat(row: int, col: int) -> tuple[float, float]:
    """Centre of a pixel of the designed grid, in lon/lat."""
    x, y = _TO_LONLAT.transform(LEFT + (col + 0.5) * 10.0, TOP - (row + 0.5) * 10.0)
    return float(x), float(y)


def chunks_read(log: KeyLog) -> set[tuple[str, int, int, int]]:
    """(layer, year, chunk row, chunk col) of every chunk key the log saw."""
    out = set()
    for key in log.chunk_keys():
        parts = key.split("/")
        if len(parts) >= 6 and parts[1] in LAYERS and parts[2] == "c":
            out.add((parts[1], YEARS[int(parts[3])], int(parts[4]), int(parts[5])))
    return out


# --------------------------------------------------------------------------------------------
# The window it reads
# --------------------------------------------------------------------------------------------


def test_a_click_reads_only_the_chunk_that_holds_it_for_the_years_needed(demo):
    store, log = demo
    log.log.clear()
    traj = run_click(store, *lonlat(96, 88))
    assert (traj.zone, traj.row, traj.col, traj.chunk) == ("10N", 96, 88, (1, 1))
    assert (traj.rows, traj.cols) == ((64, 128), (64, 128))
    assert traj.window_km == (0.64, 0.64)
    # The window's own rules with a burned and unburned split also read 2019, for the app's check
    # of the unburned reference.
    expected = {(layer, y, 1, 1) for layer in LAYERS for y in YEARS_WITH_PAIR}
    assert chunks_read(log) == expected
    assert traj.years_fetched == YEARS_WITH_PAIR and traj.years_cached == []
    assert traj.bytes_read > 0 and traj.bytes_read == store.reader.bytes_read
    assert traj.snapshot == SNAP
    assert [p.year for p in traj.points] == YEARS_WITH_PAIR
    assert traj.run is None and traj.reference_check is not None
    assert traj.wall_s >= traj.read_s > 0
    # The outline the page draws is the window's square.
    west, south, east, north = traj.window_lonlat.bounds
    assert west < traj.lon < east and south < traj.lat < north


def test_the_stable_pair_is_read_for_the_threshold_fallback(demo):
    store, log = demo
    log.log.clear()
    # Chunk (0, 0) is forest throughout: its distances are not bimodal, so the app's rule falls back
    # to the 99th percentile of 2019 to 2020 and reads 2019 for that chunk alone.
    traj = run_click(store, *lonlat(20, 20))
    assert traj.threshold is not None and traj.threshold["method"] == "stable_p99"
    assert traj.threshold["stable_pair"] == [2019, 2020]
    assert traj.years_fetched == YEARS_WITH_PAIR
    assert chunks_read(log) == {(layer, y, 0, 0) for layer in LAYERS for y in YEARS_WITH_PAIR}
    assert traj.point(2019).role == "stable pair"
    # No split, so no reference to check; the note says the fallback flags the most-changed pixels.
    assert traj.reference_check is None
    fallback = [n for n in traj.notes if n.startswith("The threshold is the app's fallback")]
    assert len(fallback) == 1 and "most-changed pixels as burned even where nothing burned" in fallback[0]
    share = traj.threshold["fallback_flagged_share"]
    assert f"Here {share:.1%} of the window's classified pixels are over it" in fallback[0]


# --------------------------------------------------------------------------------------------
# The numbers, against the wildfire app's own run on the same chunk
# --------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def full_run(scar_mem, tmp_path_factory):
    """The wildfire app on chunk (1, 1) as its AOI: what a click in that chunk must reproduce."""
    tmp = tmp_path_factory.mktemp("demo_full")
    aoi = tmp / "chunk.geojson"
    polygon = grid_polygon(LEFT, TOP, ((64, 128), (64, 128)))
    aoi.write_text(json.dumps({"type": "Feature", "properties": {}, "geometry": mapping(polygon)}))
    reader = TesseraReader.from_store(KeyLog(scar_mem), snapshot_id=SNAP)
    result = wildfire.run_wildfire(aoi, 2020, 2022, tmp / "out", reader=reader)
    assert result.summary["aoi_pixels"] == CHUNK * CHUNK
    with rasterio.open(result.paths.tif) as ds:
        bands = {name: ds.read(i + 1) for i, name in enumerate(ds.descriptions)}
        transform = ds.transform
    return result, bands, transform


def _band_at(full_run, name: str, row: int, col: int) -> float:
    _, bands, transform = full_run
    x, y = LEFT + (col + 0.5) * 10.0, TOP - (row + 0.5) * 10.0
    r, c = rowcol(transform, x, y)
    return float(bands[name][r, c])


@pytest.mark.parametrize("pixel", [(96, 88), (82, 74), (110, 102), (88, 78)])
def test_the_burned_pixel_matches_the_apps_full_run_on_its_chunk(demo, full_run, pixel):
    store, _ = demo
    result, _, _ = full_run
    summary = result.summary
    traj = run_click(store, *lonlat(*pixel))
    assert traj.threshold is not None
    assert traj.threshold["method"] == summary["threshold"]["method"] == "otsu"
    assert traj.threshold["threshold"] == summary["threshold"]["threshold"]
    assert traj.classified_pixels == summary["classified_pixels"]
    assert traj.burned_pixels == summary["burned_pixels"] == 32 * 32 - 4 * 4
    assert traj.reference_pixels == summary["recovery_reference"]["pixels"]
    assert traj.split and traj.pixel_burned
    assert traj.pixel_distance == pytest.approx(_band_at(full_run, "distance_2020_2022", *pixel), abs=1e-6)
    assert traj.point(2022).similarity == pytest.approx(1.0 - traj.pixel_distance, abs=1e-6)
    assert traj.point(2020).similarity == pytest.approx(1.0, abs=1e-6)
    for row in summary["recovery"]:
        p = traj.point(int(row["year"]))
        assert p.reference_median == pytest.approx(row["median_sim_unburned"], abs=1e-9)
        assert p.window_recovered_fraction == pytest.approx(row["recovered_fraction"], abs=1e-9)
        assert p.window_median_recovery_index == pytest.approx(row["median_recovery_index"], abs=1e-6)
        band = _band_at(full_run, f"recovery_index_{row['year']}", *pixel)
        if np.isnan(band):
            assert p.recovery_index is None
        else:
            assert p.recovery_index == pytest.approx(band, abs=1e-6)
    assert traj.point(2022).recovery_index == pytest.approx(0.0, abs=1e-6)
    # The scar returns to forest from 2023, so its recovery index climbs.
    assert traj.point(2025).recovery_index > traj.point(2023).recovery_index > 0
    # The app's check of its reference, 2019 to 2020 against 2020 to 2022, and its radar-free share.
    record = summary["recovery_reference"]
    check = traj.reference_check
    assert check is not None and check["not_run"] is None and check["years"] == [2019, 2020]
    assert check["pre_fire_median"] == pytest.approx(record["pre_fire_similarity"]["median"], abs=1e-9)
    assert check["post_fire_median"] == pytest.approx(record["post_fire_similarity"]["median"], abs=1e-9)
    assert check["drift"] == pytest.approx(record["pre_minus_post_median"], abs=1e-9)
    assert check["drift"] <= wildfire.REFERENCE_DRIFT_WARN
    radar = summary["radar"]
    assert traj.radar_free_scar == pytest.approx(
        (radar["radar_free_fraction_in_scar_year_a"], radar["radar_free_fraction_in_scar_year_b"]), abs=1e-12
    )
    assert not radar["radar_free_flag"]
    assert not any("drift" in n or "less similar to itself" in n or "radar-free" in n for n in traj.notes)


def test_the_numbers_are_the_apps_functions_on_the_window(demo):
    store, _ = demo
    traj = run_click(store, *lonlat(96, 88))
    reader = store.reader
    rows, cols = slice(64, 128), slice(64, 128)
    wins = reader.read_windows("10N", YEARS_DEFAULT, rows, cols, layers=wildfire.LAYERS)
    lr, lc = 96 - 64, 88 - 64
    a = wins[2020]
    for y in YEARS_DEFAULT:
        expected = float(metrics.cosine_similarity_map(a, wins[y])[lr, lc])
        assert traj.point(y).similarity == pytest.approx(expected, abs=1e-6)
    s_post = 1.0 - traj.pixel_distance
    for y in (2023, 2024, 2025):
        p = traj.point(y)
        r = wildfire.recovery_index(np.float32(p.similarity), np.float32(s_post), p.reference_median)
        assert p.recovery_index == pytest.approx(float(r), abs=1e-6)


def test_a_radar_switch_is_flagged_and_left_out_as_the_app_does(demo, full_run):
    store, _ = demo
    traj = run_click(store, *lonlat(87, 77))  # inside the 2024 radar gap
    p = traj.point(2024)
    assert p.radar is False and p.radar_matches is False and not p.comparable
    assert p.similarity is not None  # shown, drawn hollow
    assert p.recovery_index is None
    assert np.isnan(_band_at(full_run, "recovery_index_2024", 87, 77))
    assert traj.point(2023).recovery_index is not None
    radar = [n for n in traj.notes if "Sentinel-1" in n]
    assert len(radar) == 1 and "2020 and 2024" in radar[0] and "survey B5" in radar[0]


def test_a_pixel_whose_post_fire_year_switches_radar_is_not_classified(demo):
    store, _ = demo
    traj = run_click(store, *lonlat(101, 91))  # inside the 2022 radar gap
    assert traj.pixel_distance is None and traj.pixel_burned is None
    assert traj.split  # the window still has its split
    assert all(p.recovery_index is None for p in traj.points)
    assert traj.point(2023).similarity is not None
    assert any("is not classified" in n for n in traj.notes)


def test_an_unburned_pixel_says_so_and_still_has_its_curve(demo):
    store, _ = demo
    traj = run_click(store, *lonlat(66, 120))  # forest in the scar's chunk
    assert traj.split and traj.pixel_burned is False
    assert all(p.recovery_index is None for p in traj.points)
    assert traj.point(2023).reference_median is not None
    assert all(p.similarity is not None for p in traj.points)
    assert any(n.startswith("This pixel did not burn") for n in traj.notes)


def test_a_window_with_no_split_says_so_and_still_has_its_curve(demo):
    store, _ = demo
    traj = run_click(store, *lonlat(20, 20))
    assert not traj.split and traj.burned_pixels == 0
    assert all(p.recovery_index is None and p.reference_median is None for p in traj.points)
    assert all(p.similarity is not None for p in traj.points)
    assert any("no burned and unburned split" in n for n in traj.notes)


def test_every_click_with_the_windows_own_rules_says_so(demo):
    store, _ = demo
    traj = run_click(store, *lonlat(96, 88))
    assert traj.notes[-1] == trajectory.REFERENCE_NOTE.format(size="0.64 km by 0.64 km", such_as="")
    assert "threshold, the burned verdict and the unburned reference here are this window's own" in traj.notes[-1]
    assert "can differ from a run over a whole fire" in traj.notes[-1]


def test_the_roles_name_the_fire_year_alone():
    roles = {y: trajectory._role(y, 2019, 2022, wildfire.default_fire_year(2019, 2022)) for y in range(2018, 2026)}
    assert roles == {
        2018: "stable pair",
        2019: "pre-fire",
        2020: "before the fire",
        2021: "fire",
        2022: "post-fire",
        2023: "recovery",
        2024: "recovery",
        2025: "recovery",
    }
    # year_b right after year_a: year_b is the fire year, and no year lies between.
    assert [trajectory._role(y, 2020, 2021, 2021) for y in (2020, 2021, 2022)] == ["pre-fire", "post-fire", "recovery"]


def test_a_click_with_years_around_the_fire_labels_only_the_fire_year_fire(demo):
    store, _ = demo
    traj = run_click(store, *lonlat(96, 88), year_a=2019, year_b=2022)
    assert traj.fire_year == 2021
    assert traj.point(2020).role == "before the fire" and traj.point(2021).role == "fire"
    assert [p.year for p in traj.points if p.role == "fire"] == [2021]


# --------------------------------------------------------------------------------------------
# The app's warnings about the window's own threshold and reference
# --------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def warning_mem() -> MemoryStore:
    return _mem(pattern=_warning_pattern, prototypes=_warning_prototypes)


def _app_run(mem: MemoryStore, box_rc, tmp: Path, **kwargs):
    aoi = tmp / "aoi.geojson"
    aoi.write_text(
        json.dumps({"type": "Feature", "properties": {}, "geometry": mapping(grid_polygon(LEFT, TOP, box_rc))})
    )
    reader = TesseraReader.from_store(KeyLog(mem), snapshot_id=SNAP)
    return wildfire.run_wildfire(aoi, 2020, 2022, tmp / "out", reader=reader, **kwargs)


def test_a_drifted_reference_and_a_radar_free_scar_are_flagged_as_the_app_flags_them(warning_mem, tmp_path):
    result = _app_run(warning_mem, CHUNK_11, tmp_path)
    summary, warnings = result.summary, result.report["warnings"]
    store = DemoStore.from_reader(TesseraReader.from_store(KeyLog(warning_mem), snapshot_id=SNAP))
    traj = run_click(store, *lonlat(96, 88))
    assert traj.threshold["threshold"] == summary["threshold"]["threshold"]
    assert (
        traj.burned_pixels == summary["burned_pixels"]
        and traj.reference_pixels == summary["recovery_reference"]["pixels"]
    )
    record = summary["recovery_reference"]
    assert record["pre_minus_post_median"] > wildfire.REFERENCE_DRIFT_WARN
    assert traj.reference_check["drift"] == pytest.approx(record["pre_minus_post_median"], abs=1e-9)
    assert any("less similar to itself after the fire" in w for w in warnings)
    drift = [n for n in traj.notes if "less similar to itself after the fire" in n]
    assert len(drift) == 1
    assert f"(median {record['post_fire_similarity']['median']:.3f}, 2020 to 2022)" in drift[0]
    assert f"({record['pre_fire_similarity']['median']:.3f}, 2019 to 2020)" in drift[0]
    assert "inflates the recovery index here" in drift[0] and "Extend the AOI" not in drift[0]
    radar = summary["radar"]
    assert radar["radar_free_flag"] and any("radar-free" in w for w in warnings)
    assert traj.radar_free_scar == pytest.approx(
        (radar["radar_free_fraction_in_scar_year_a"], radar["radar_free_fraction_in_scar_year_b"]), abs=1e-12
    )
    flagged = [n for n in traj.notes if "radar-free" in n]
    assert len(flagged) == 1 and "More than 20% of the window's scar is radar-free" in flagged[0]


def test_a_suspect_fallback_is_flagged_as_the_app_flags_it(warning_mem, tmp_path):
    result = _app_run(warning_mem, CHUNK_22, tmp_path)
    choice = result.summary["threshold"]
    assert choice["method"] == "stable_p99" and choice["fallback_suspect"]
    store = DemoStore.from_reader(TesseraReader.from_store(KeyLog(warning_mem), snapshot_id=SNAP))
    traj = run_click(store, *lonlat(160, 160))
    assert traj.threshold["threshold"] == choice["threshold"] and traj.threshold["fallback_suspect"]
    app_text = next(w for w in result.report["warnings"] if "stable pair (2019, 2020)" in w)
    notes = [n for n in traj.notes if n.startswith("Threshold: the stable pair")]
    assert len(notes) == 1
    # The app's own words, without its advice to pass --threshold, which the page has no input for.
    assert app_text.startswith(notes[0][:-1])
    assert "--threshold" not in notes[0] and "--threshold" in app_text
    assert not any("most-changed pixels" in n for n in traj.notes)


def test_the_reference_check_says_when_it_could_not_run(demo):
    store, _ = demo
    traj = run_click(store, *lonlat(96, 88), year_a=2017)  # 2016 is before the store's first year
    assert traj.split and traj.reference_check["drift"] is None
    assert traj.reference_check["not_run"] == "2016 is before 2017, the store's first year"
    assert any("was not run on the window's: 2016 is before 2017" in n for n in traj.notes)
    log = KeyLog(_mem([y for y in YEARS if y != 2019]))
    gap = DemoStore.from_reader(TesseraReader.from_store(log, snapshot_id=SNAP))
    traj = run_click(gap, *lonlat(96, 88))
    assert traj.split and traj.years_fetched == YEARS_DEFAULT
    assert traj.reference_check["not_run"] == "zone 10N has not published 2019"


# --------------------------------------------------------------------------------------------
# A precomputed run's threshold and reference, for the clicks it covers
# --------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def grid_run(scar_mem, tmp_path_factory):
    """The wildfire app on the whole grid: a full run over the fire, as the precomputed Caldor run is."""
    tmp = tmp_path_factory.mktemp("demo_grid")
    result = _app_run(scar_mem, WHOLE_GRID, tmp)
    with rasterio.open(result.paths.tif) as ds:
        bands = {name: ds.read(i + 1) for i, name in enumerate(ds.descriptions)}
        transform = ds.transform
    rows = {int(r["year"]): r for r in result.summary["recovery"]}
    settings = trajectory.RunSettings(
        name="grid",
        snapshot=SNAP,
        year_a=2020,
        year_b=2022,
        geometry=grid_polygon(LEFT, TOP, WHOLE_GRID),
        threshold=float(result.summary["threshold"]["threshold"]),
        method=result.summary["threshold"]["method"],
        reference_median={y: float(r["median_sim_unburned"]) for y, r in rows.items()},
        reference_p10={y: float(r["unburned_sim_p10"]) for y, r in rows.items()},
        reference_pixels={y: int(r["n_unburned"]) for y, r in rows.items()},
        source="the test's full run",
    )
    return result, bands, transform, settings


@pytest.fixture
def run_store(scar_mem, grid_run) -> tuple[DemoStore, KeyLog]:
    log = KeyLog(scar_mem)
    return DemoStore.from_reader(TesseraReader.from_store(log, snapshot_id=SNAP), runs=[grid_run[3]]), log


@pytest.mark.parametrize("pixel", [(96, 88), (82, 74), (110, 102), (88, 78), (87, 77)])
def test_a_click_a_run_covers_takes_its_threshold_and_reference(run_store, grid_run, demo, pixel):
    store, log = run_store
    result, _, _, settings = grid_run
    log.log.clear()
    traj = run_click(store, *lonlat(*pixel))
    full = (None, *grid_run[1:3])
    assert traj.run is not None and traj.run["name"] == "grid" and traj.run["threshold"] == settings.threshold
    assert traj.threshold["method"] == "override" and traj.threshold["threshold"] == settings.threshold
    # No fallback and no reference of the window's own: 2019 is not read.
    assert traj.years_fetched == YEARS_DEFAULT and traj.reference_check is None
    assert {k[1] for k in chunks_read(log)} == set(YEARS_DEFAULT)
    assert traj.reference_pixels == settings.reference_pixels[2022]
    # The scar lies inside the window, so the window's burned pixels are the run's, and the pixel's
    # verdict and recovery index are the run's bands.
    assert traj.burned_pixels == result.summary["burned_pixels"]
    assert traj.pixel_burned == bool(_band_at(full, "burned", *pixel))
    for row in result.summary["recovery"]:
        p = traj.point(int(row["year"]))
        assert p.reference_median == row["median_sim_unburned"]
        assert p.window_recovered_fraction == pytest.approx(row["recovered_fraction"], abs=1e-12)
        assert p.window_median_recovery_index == pytest.approx(row["median_recovery_index"], abs=1e-6)
        band = _band_at(full, f"recovery_index_{row['year']}", *pixel)
        assert p.recovery_index is None if np.isnan(band) else p.recovery_index == pytest.approx(band, abs=1e-6)
    assert traj.notes[-1].startswith("This pixel is inside the Grid demo AOI at the run's snapshot and years")
    assert "the precomputed Grid run's threshold" in traj.notes[-1]
    # The window's own reference is another, so its recovery index is another (the threshold happens
    # to be the same here: Otsu splits the same gap between ash and forest).
    own = run_click(demo[0], *lonlat(*pixel))
    assert own.run is None and own.reference_pixels != traj.reference_pixels
    for year in (2023, 2024, 2025):
        assert own.point(year).reference_median != traj.point(year).reference_median


def test_a_quiet_window_a_run_covers_has_no_burn_where_its_own_fallback_would_flag_some(run_store, grid_run, demo):
    store, _ = run_store
    traj = run_click(store, *lonlat(20, 20))  # chunk (0, 0): forest throughout
    assert traj.threshold["method"] == "override" and traj.burned_pixels == 0 and not traj.split
    assert traj.pixel_burned is False and _band_at((None, *grid_run[1:3]), "burned", 20, 20) == 0.0
    assert traj.years_fetched == YEARS_DEFAULT
    assert any("above the threshold" in n and "the precomputed Grid run's, by Otsu" in n for n in traj.notes)
    own = run_click(demo[0], *lonlat(20, 20))
    assert own.threshold["method"] == "stable_p99"


def test_the_runs_threshold_is_the_one_applied(scar_mem, grid_run):
    settings = dataclasses.replace(grid_run[3], threshold=1.5)
    store = DemoStore.from_reader(TesseraReader.from_store(KeyLog(scar_mem), snapshot_id=SNAP), runs=[settings])
    traj = run_click(store, *lonlat(96, 88))
    assert traj.threshold["threshold"] == 1.5 and traj.threshold["method"] == "override"
    assert traj.burned_pixels == 0 and traj.pixel_burned is False and not traj.split
    assert any("above the threshold 1.500" in n for n in traj.notes)


@pytest.mark.parametrize(
    ("kwargs", "where"),
    [
        ({"year_a": 2019}, (96, 88)),
        ({"year_b": 2023}, (96, 88)),
        ({}, None),
    ],
)
def test_a_click_the_run_does_not_cover_uses_the_windows_own_rules(scar_mem, grid_run, kwargs, where):
    settings = grid_run[3]
    if where is None:  # inside the grid but outside the run's AOI
        settings = dataclasses.replace(settings, geometry=grid_polygon(LEFT, TOP, ((0, 64), (0, 64))))
        where = (96, 88)
    store = DemoStore.from_reader(TesseraReader.from_store(KeyLog(scar_mem), snapshot_id=SNAP), runs=[settings])
    traj = run_click(store, *lonlat(*where), **kwargs)
    assert traj.run is None and traj.threshold["method"] != "override"
    assert traj.notes[-1].startswith("The threshold, the burned verdict and the unburned reference here are")
    assert "such as the precomputed Grid run (threshold" in traj.notes[-1]


def test_a_run_covers_only_its_snapshot_years_and_aoi(grid_run):
    settings = grid_run[3]
    lon, lat = lonlat(96, 88)
    assert settings.covers(SNAP, lon, lat, year_a=2020, year_b=2022, through=2025)
    assert settings.covers(SNAP, lon, lat, year_a=2020, year_b=2022, through=2023)
    assert not settings.covers("OTHER", lon, lat, year_a=2020, year_b=2022, through=2025)
    assert not settings.covers(SNAP, lon, lat, year_a=2019, year_b=2022, through=2025)
    assert not settings.covers(SNAP, lon, lat, year_a=2020, year_b=2023, through=2025)
    assert not settings.covers(SNAP, lon + 0.1, lat, year_a=2020, year_b=2022, through=2025)
    short = dataclasses.replace(settings, reference_median={2022: 0.9, 2023: 0.9})
    assert not short.covers(SNAP, lon, lat, year_a=2020, year_b=2022, through=2025)


def test_the_precomputed_caldor_runs_settings_are_read_from_demo_precomputed():
    settings = trajectory.load_run_settings()
    index = json.loads((trajectory.PRECOMPUTED / "index.json").read_text(encoding="utf-8"))
    caldor = next(d for d in index["demos"] if d["name"] == "caldor")
    assert settings.name == "caldor" and settings.label == "the precomputed Caldor run"
    assert settings.snapshot == trajectory.SNAPSHOT == index["snapshot"]
    assert (settings.year_a, settings.year_b) == (2020, 2022) and settings.method == "otsu"
    assert settings.threshold == 0.24403303861618042
    assert settings.threshold == next(h["value"] for h in caldor["headline"] if h["key"] == "threshold")
    assert settings.reference_median == {
        2022: 0.9639418721199036,
        2023: 0.9582132697105408,
        2024: 0.9698836803436279,
        2025: 0.9603413939476013,
    }
    assert settings.reference_pixels[2022] == 19488562
    # A burned pixel of the full run, and a point outside the AOI.
    assert settings.covers(trajectory.SNAPSHOT, -120.22593, 38.71735, year_a=2020, year_b=2022, through=2025)
    assert not settings.covers(trajectory.SNAPSHOT, -121.5, 38.7, year_a=2020, year_b=2022, through=2025)


def test_missing_precomputed_files_are_refused_with_what_to_do(tmp_path):
    with pytest.raises(ValueError, match="Restore demo/precomputed from the repository") as info:
        trajectory.load_run_settings(tmp_path)
    assert "FileNotFoundError" in str(info.value)
    (tmp_path / "index.json").write_text(json.dumps({"demos": [{"name": "caldor", "app": "deforest"}]}))
    with pytest.raises(ValueError, match="not a wildfire run"):
        trajectory.load_run_settings(tmp_path)


# --------------------------------------------------------------------------------------------
# A click on another copy of the world
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("turns", [1, -1, 2])
def test_a_longitude_on_another_copy_of_the_world_is_wrapped(demo, turns):
    store, _ = demo
    lon, lat = lonlat(96, 88)
    here = run_click(store, lon, lat)
    there = run_click(store, lon + 360.0 * turns, lat)
    assert (there.zone, there.row, there.col, there.chunk) == (here.zone, here.row, here.col, here.chunk)
    assert there.lon == pytest.approx(lon, abs=1e-9) and there.points == here.points


def test_wrap_longitude():
    assert trajectory.wrap_longitude(202.1402) == pytest.approx(-157.8598, abs=1e-9)
    assert trajectory.wrap_longitude(-200.0) == pytest.approx(160.0)
    assert trajectory.wrap_longitude(180.0) == 180.0 and trajectory.wrap_longitude(-180.0) == -180.0
    assert trajectory.wrap_longitude(-120.5) == -120.5
    assert math.isnan(trajectory.wrap_longitude(float("nan")))
    assert trajectory.wrap_longitude(float("inf")) == float("inf")


# --------------------------------------------------------------------------------------------
# The cache
# --------------------------------------------------------------------------------------------


def test_a_second_click_in_the_same_chunk_and_years_reads_nothing(demo):
    store, log = demo
    first = run_click(store, *lonlat(96, 88))
    log.log.clear()
    before = store.reader.bytes_read
    second = run_click(store, *lonlat(70, 70))
    assert second.chunk == first.chunk
    assert second.bytes_read == 0 and store.reader.bytes_read == before
    assert second.from_cache and second.years_cached == YEARS_WITH_PAIR
    assert store.clicks == 2
    assert chunks_read(log) == set()
    # Same pixel again: the same numbers.
    again = run_click(store, *lonlat(96, 88))
    assert again.from_cache and again.points == first.points


def test_new_years_read_only_the_years_not_yet_cached(demo):
    store, log = demo
    run_click(store, *lonlat(96, 88))
    log.log.clear()
    # 2019 to 2025 are cached; the new stable pair needs 2018 alone.
    traj = run_click(store, *lonlat(96, 88), year_a=2019)
    assert traj.years_fetched == [2018]
    assert traj.years_cached == YEARS_WITH_PAIR
    assert chunks_read(log) == {(layer, 2018, 1, 1) for layer in LAYERS}


def test_another_chunk_is_read_afresh(demo):
    store, log = demo
    run_click(store, *lonlat(96, 88))
    log.log.clear()
    traj = run_click(store, *lonlat(96, 150))
    assert traj.chunk == (1, 2) and traj.bytes_read > 0
    assert {k[2:] for k in chunks_read(log)} == {(1, 2)}


def test_the_cache_is_keyed_by_snapshot_and_drops_the_oldest_past_its_size(demo):
    store, _ = demo
    run_click(store, *lonlat(96, 88))
    win = store.cache.get((SNAP, "10N", 1, 1, 2020))
    assert win is not None
    assert store.cache.get(("OTHER", "10N", 1, 1, 2020)) is None
    size = trajectory._window_nbytes(win)
    small = WindowCache(max_bytes=2 * size)
    for year in (2020, 2021, 2022):
        small.put((SNAP, "10N", 1, 1, year), win)
    assert len(small) == 2 and small.nbytes == 2 * size
    assert small.get((SNAP, "10N", 1, 1, 2020)) is None
    assert small.get((SNAP, "10N", 1, 1, 2022)) is win
    assert WindowCache(max_bytes=0).max_bytes == 0
    empty = WindowCache(max_bytes=0)
    empty.put((SNAP, "10N", 1, 1, 2020), win)
    assert len(empty) == 0


def test_a_zero_size_cache_reads_every_click(scar_mem):
    reader = TesseraReader.from_store(KeyLog(scar_mem), snapshot_id=SNAP)
    store = DemoStore.from_reader(reader, cache=WindowCache(max_bytes=0))
    first = run_click(store, *lonlat(96, 88))
    second = run_click(store, *lonlat(96, 88))
    third = run_click(store, *lonlat(96, 88))
    # The first read of a zone also builds geotessera's view of it (its time coordinate, a few bytes).
    assert 0 < second.bytes_read == third.bytes_read <= first.bytes_read
    assert second.years_fetched == first.years_fetched == YEARS_WITH_PAIR
    assert second.points == first.points


def test_cache_size_and_snapshot_come_from_the_environment():
    assert trajectory.cache_bytes({}) == trajectory.DEFAULT_CACHE_MB * 2**20
    assert trajectory.cache_bytes({trajectory.CACHE_MB_ENV: "64"}) == 64 * 2**20
    with pytest.raises(ValueError, match="whole number of MB"):
        trajectory.cache_bytes({trajectory.CACHE_MB_ENV: "lots"})
    with pytest.raises(ValueError, match="negative"):
        trajectory.cache_bytes({trajectory.CACHE_MB_ENV: "-1"})
    assert trajectory.resolve_snapshot({}) == trajectory.SNAPSHOT == "RC6FS07ZB0T76EXT8XRG"
    assert trajectory.resolve_snapshot({trajectory.SNAPSHOT_ENV: " ABC "}) == "ABC"
    assert trajectory.resolve_snapshot({trajectory.SNAPSHOT_ENV: ""}) == trajectory.SNAPSHOT


def test_open_store_pins_the_snapshot_and_explains_a_store_it_cannot_open(monkeypatch, scar_mem, tmp_path):
    seen: list[str | None] = []

    def fake_open(*, snapshot_id=None, branch="main", cache=None):
        seen.append(snapshot_id)
        if snapshot_id == "MISSING":
            raise StoreUnavailableError("Cannot open the store at snapshot MISSING: not found.")
        return TesseraReader.from_store(KeyLog(scar_mem), snapshot_id=snapshot_id)

    monkeypatch.setattr(trajectory.TesseraReader, "open", staticmethod(fake_open))
    monkeypatch.delenv(trajectory.SNAPSHOT_ENV, raising=False)
    opened = trajectory.open_store()
    assert opened.snapshot == trajectory.SNAPSHOT
    assert [r.name for r in opened.runs] == ["caldor"]
    monkeypatch.setenv(trajectory.SNAPSHOT_ENV, "OTHERSNAP")
    assert trajectory.open_store().snapshot == "OTHERSNAP"
    with pytest.raises(ClickError) as info:
        trajectory.open_store("MISSING")
    assert info.value.kind == "store"
    assert "MISSING" in str(info.value) and trajectory.SNAPSHOT_ENV in str(info.value)
    assert seen == [trajectory.SNAPSHOT, "OTHERSNAP", "MISSING"]
    # Missing precomputed files stop the store opening, before the store is asked for anything.
    with pytest.raises(ClickError) as info:
        trajectory.open_store(precomputed=tmp_path)
    assert info.value.kind == "store" and "Restore demo/precomputed" in str(info.value)
    assert seen == [trajectory.SNAPSHOT, "OTHERSNAP", "MISSING"]


def test_a_reader_without_a_snapshot_is_refused(scar_mem):
    with pytest.raises(ValueError, match="snapshot_id"):
        DemoStore.from_reader(TesseraReader.from_store(KeyLog(scar_mem)))


# --------------------------------------------------------------------------------------------
# Failures: each one says what to change, and only no-data needs a read
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("lon", "lat", "why"),
    [
        (10.0, 45.0, "the store has no zone group 32N"),
        (-123.5, 38.8, "falls off zone group 10N's grid"),
        (-120.4, 85.0, "no polar zones"),
        (200.0, 38.8, "the store has no zone group 04N"),  # wrapped to -160
        (float("nan"), 38.8, "longitude is not between -180 and 180"),
    ],
)
def test_a_click_outside_the_store_says_where_to_click_before_any_read(demo, lon, lat, why):
    store, log = demo
    log.log.clear()
    with pytest.raises(ClickError) as info:
        run_click(store, lon, lat)
    assert info.value.kind == "outside"
    assert info.value.bytes_read == 0 and info.value.wall_s >= 0
    message = str(info.value)
    assert "outside the store's coverage" in message and why in message
    assert "Click on land" in message
    assert log.chunk_keys() == []


def test_a_pixel_with_no_data_says_click_on_land(demo):
    store, log = demo
    log.log.clear()
    with pytest.raises(ClickError) as info:
        run_click(store, *lonlat(30, 160))  # the open water in chunk (0, 2)
    assert info.value.kind == "no_data"
    message = str(info.value)
    assert "no data" in message and "in 2020, the pre-fire year" in message
    assert "Click on land, or choose another pre-fire year." in message
    # It had to read the chunk to know: the error carries what the click cost.
    assert {k[2:] for k in chunks_read(log)} == {(0, 2)}
    assert info.value.bytes_read == store.reader.bytes_read
    assert info.value.wall_s > 0


def test_a_pixel_with_no_post_fire_data_says_which_year_to_change(demo):
    store, _ = demo
    with pytest.raises(ClickError) as info:
        run_click(store, *lonlat(155, 15))
    assert info.value.kind == "no_data"
    assert "in 2022, the post-fire year" in str(info.value)
    assert "choose another post-fire year" in str(info.value)
    assert info.value.bytes_read > 0  # the rest of the chunk has data
    # Another post-fire year works from the cached years; the chunk is all forest, so the threshold
    # falls back to the stable pair, whose 2019 is the one year read.
    traj = run_click(store, *lonlat(155, 15), year_b=2023)
    assert traj.years_fetched == [2019] and traj.years_cached == YEARS_DEFAULT
    assert traj.point(2022).similarity is None and traj.point(2023).similarity is not None


def test_a_year_without_data_is_skipped_and_named(demo):
    store, _ = demo
    traj = run_click(store, *lonlat(175, 15))
    assert traj.point(2024).similarity is None and not traj.point(2024).comparable
    assert traj.point(2025).similarity is not None
    assert any("no data at this pixel in 2024" in n for n in traj.notes)


def test_years_the_zone_has_not_published_are_refused_before_any_read(tmp_path):
    published = [y for y in YEARS if y != 2023]
    log = KeyLog(_mem(published))
    store = DemoStore.from_reader(TesseraReader.from_store(log, snapshot_id=SNAP))
    log.log.clear()
    with pytest.raises(ClickError) as info:
        run_click(store, *lonlat(96, 88))
    assert info.value.kind == "years"
    message = str(info.value)
    assert message.startswith(
        "Zone 10N has not published 2023 (it publishes 2017, 2018, 2019, 2020, 2021, 2022, 2024 and 2025). The click "
        "reads every year from the pre-fire year (2020) to the last recovery year (2025)"
    )
    assert "With the last recovery year 2025, a pre-fire year of 2024 works here." in message
    assert (
        "Or keep the pre-fire year 2020 and the post-fire year 2022 and set the last recovery year to 2022." in message
    )
    assert message.endswith("Change the years above the map, or click in another zone.")
    assert "AOI" not in message and "year_a" not in message
    assert log.chunk_keys() == []
    # Both ways out work.
    traj = run_click(store, *lonlat(96, 88), year_a=2020, year_b=2022, through=2022)
    assert [p.year for p in traj.points] == [2019, 2020, 2021, 2022]
    traj = run_click(store, *lonlat(96, 88), year_a=2024, year_b=2025)
    assert [p.year for p in traj.points if p.role != "stable pair"] == [2024, 2025]


def test_a_zone_with_a_gap_after_the_first_years_names_the_pre_fire_years_that_work():
    # As zone 03S at Pukapuka: 2017, 2018, 2020 and 2021 are not published.
    log = KeyLog(_mem([2019, 2022, 2023, 2024, 2025]))
    store = DemoStore.from_reader(TesseraReader.from_store(log, snapshot_id=SNAP))
    log.log.clear()
    for year_a in (2020, 2019):
        with pytest.raises(ClickError) as info:
            run_click(store, *lonlat(96, 88), year_a=year_a, year_b=2022)
        message = str(info.value)
        assert info.value.kind == "years" and info.value.bytes_read == 0
        assert "it publishes 2019, 2022, 2023, 2024 and 2025" in message
        assert "a pre-fire year of 2022, 2023 or 2024 works here." in message
        assert "Or keep the pre-fire year" not in message
    assert log.chunk_keys() == []
    # One of the years named does work; its stable pair 2021 is not published, so the check of the
    # window's reference, when there is one, says it was not run.
    traj = run_click(store, *lonlat(96, 88), year_a=2022, year_b=2023)
    assert traj.years_fetched == [2022, 2023, 2024, 2025]
    if traj.reference_check is not None:
        assert traj.reference_check["not_run"] == "zone 10N has not published 2021"


@pytest.mark.parametrize(
    ("kwargs", "text"),
    [
        ({"year_a": 2022, "year_b": 2022}, "The post-fire year (2022) must be after the pre-fire year (2022)"),
        ({"year_a": 2016, "year_b": 2018}, "Years must lie in 2017 to 2025; got the pre-fire year 2016"),
        (
            {"year_a": 2020, "year_b": 2022, "through": 2021},
            "The last recovery year (2021) must lie between the post-fire year (2022) and 2025",
        ),
    ],
)
def test_bad_years_are_refused_in_the_pages_words(demo, kwargs, text):
    store, log = demo
    log.log.clear()
    with pytest.raises(ClickError) as info:
        run_click(store, *lonlat(96, 88), **kwargs)
    message = str(info.value)
    assert info.value.kind == "years"
    assert text in message and "Change the years above the map." in message
    assert not any(name in message for name in ("year_a", "year_b", "recovery_through"))
    assert log.chunk_keys() == []


def test_the_apps_argument_names_are_put_in_the_pages_words():
    text = "recovery_through (2021) must lie between year_b (2022) and 2025; fire_year (2021) after year_a (2020)."
    assert trajectory._page_words(text) == (
        "the last recovery year (2021) must lie between the post-fire year (2022) and 2025; the fire year (2021) "
        "after the pre-fire year (2020)."
    )


def test_the_module_imports_no_streamlit():
    code = "import sys, demo.trajectory; sys.exit(1 if 'streamlit' in sys.modules else 0)"
    done = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, check=False)
    assert done.returncode == 0, done.stderr


def test_messages_hold_no_dashes():
    text = Path(trajectory.__file__).read_text(encoding="utf-8")
    assert "\u2014" not in text and "\u2013" not in text
