"""Geometry tests for tessera_apps.io: band and equator edges, grid origins, zone limits and plan size guards.

Planning tests use :class:`LiveGridReader`, a stub whose grids carry the live store's metadata (north
groups end at northing 0, south groups start at 10,014,720), so they read no chunk. The equator point
tests use in-memory zone groups with the live dimensions, written only near the equator. No network.
"""

from __future__ import annotations

import tracemalloc
from collections.abc import Sequence
from typing import Any

import numpy as np
import pytest
import shapely
import zarr
from pyproj import Geod, Transformer
from shapely.geometry import LineString, MultiPolygon, Polygon, box
from shapely.geometry.base import BaseGeometry
from zarr.storage import MemoryStore

import tessera_apps.io as tio
from tessera_apps.io import (
    DEFAULT_LAYERS,
    MAX_AOI_KM2,
    N_BANDS,
    YEARS,
    AOIError,
    AOIPlan,
    TesseraError,
    TesseraReader,
    Window,
    ZoneGrid,
    ZonePlan,
    plan_aoi,
    read_points,
    zone_band,
    zone_epsg,
    zone_for,
    zone_number,
)

# ---- live-shaped grids -----------------------------------------------------------------------

#: Live grid metadata (survey 'Grid' section): every group shares left and width; north groups end
#: at northing 0 and south groups start 14.72 km north of the equator.
LIVE_LEFT = 163_840.0
LIVE_WIDTH = 67_584
LIVE = {"N": (9_338_880.0, 933_888), "S": (10_014_720.0, 890_880)}


def live_attrs(zone: str, registration: str = "corner") -> dict[str, Any]:
    """Zone attributes as the live store writes them, with the transform at the corner or the centre.

    Snapshots before the metadata fix stored the transform at the pixel centre and the same bbox.
    """
    top, height = LIVE[zone[-1]]
    shift = 5.0 if registration == "centre" else 0.0
    return {
        "proj:code": f"EPSG:{zone_epsg(zone)}",
        "spatial:transform": [10.0, 0.0, LIVE_LEFT + shift, 0.0, -10.0, top - shift],
        "spatial:bbox": [LIVE_LEFT, top - height * 10.0, LIVE_LEFT + LIVE_WIDTH * 10.0, top],
    }


def live_grid(zone: str) -> ZoneGrid:
    return ZoneGrid.from_attrs(zone, live_attrs(zone), (LIVE[zone[-1]][1], LIVE_WIDTH))


class LiveGridReader:
    """Metadata-only reader with live-shaped grids. Any chunk read fails the test."""

    def __init__(self) -> None:
        self.grid_calls: list[str] = []

    @property
    def bytes_read(self) -> int:
        return 0

    def grid(self, zone: str) -> ZoneGrid:
        self.grid_calls.append(zone)
        return live_grid(zone)

    def chunk_shape(self, zone: str) -> tuple[int, int]:
        return 256, 256

    def years_complete(self, zone: str) -> list[int]:
        return list(YEARS)

    def read_windows(
        self, zone: str, years: Sequence[int], rows: slice, cols: slice, *, layers: Sequence[str] = DEFAULT_LAYERS
    ) -> dict[int, Window]:
        raise AssertionError(f"unexpected chunk read in {zone}")

    def read_window(
        self, zone: str, year: int, rows: slice, cols: slice, *, layers: Sequence[str] = DEFAULT_LAYERS
    ) -> Window:
        raise AssertionError(f"unexpected chunk read in {zone}")

    def provenance(self) -> dict[str, Any]:
        return {}


@pytest.fixture
def live() -> LiveGridReader:
    return LiveGridReader()


def mask_centres_lonlat(p: ZonePlan) -> tuple[np.ndarray, np.ndarray]:
    """Lon/lat of the centre of every masked pixel of a zone plan."""
    rr, cc = np.nonzero(p.mask)
    xs = p.grid.left + (p.cols.start + cc + 0.5) * p.grid.pixel
    ys = p.grid.top - (p.rows.start + rr + 0.5) * p.grid.pixel
    lons, lats = p.grid.to_lonlat().transform(xs, ys)
    return np.asarray(lons), np.asarray(lats)


def assert_centres_inside(plan: AOIPlan, aoi: BaseGeometry) -> None:
    """Every masked pixel centre lies inside the AOI and inside its own zone's nominal band.

    A band edge belongs to the band east of it, and no centre sits exactly on the equator.
    """
    for p in plan.zones:
        lons, lats = mask_centres_lonlat(p)
        west, _, east, _ = zone_band(p.zone).bounds
        assert (lons >= west).all(), f"{p.zone} has {(lons < west).sum()} pixel centres west of {west}"
        assert (lons < east).all(), f"{p.zone} has {(lons >= east).sum()} pixel centres east of {east}"
        if p.zone.endswith("N"):
            assert (lats > 0).all(), f"{p.zone} has {(lats <= 0).sum()} pixel centres south of the equator"
        else:
            assert (lats < 0).all(), f"{p.zone} has {(lats >= 0).sum()} pixel centres north of the equator"
        inside = shapely.contains_xy(aoi.buffer(1e-7), lons, lats)
        assert inside.all(), f"{p.zone} has {(~inside).sum()} pixel centres outside the AOI"


def zone_counts(plan: AOIPlan) -> dict[str, int]:
    return {p.zone: p.n_pixels for p in plan.zones}


# ---- G1: AOI edges on a band edge or the equator --------------------------------------------

#: Valid L-shaped AOI from the review: its edge (-120, 38.845)-(-120, 38.85) lies on the 10N/11N line.
L_SHAPE = [
    (-120.005, 38.845),
    (-120.0, 38.845),
    (-120.0, 38.85),
    (-119.995, 38.85),
    (-119.995, 38.855),
    (-120.005, 38.855),
]


def test_l_shape_with_an_edge_on_the_band_edge_counts_each_pixel_once(boundary_reader):
    aoi = Polygon(L_SHAPE)
    assert aoi.is_valid
    # The 11N piece really is a polygon plus a line along -120: the case the fix has to handle.
    assert aoi.intersection(zone_band("11N")).geom_type == "GeometryCollection"
    # The same AOI with that edge moved 1e-9 degrees west, off the band edge.
    nudged = Polygon([(x - 1e-9 if x == -120.0 else x, y) for x, y in L_SHAPE])
    plan = plan_aoi(boundary_reader, aoi)
    ref = plan_aoi(boundary_reader, nudged)
    assert plan.zone_names == ["10N", "11N"]
    assert zone_counts(plan) == zone_counts(ref)
    assert plan.n_pixels == ref.n_pixels
    assert_centres_inside(plan, aoi)
    area_m2 = abs(Geod(ellps="WGS84").geometry_area_perimeter(aoi)[0])
    assert plan.n_pixels * 100 == pytest.approx(area_m2, rel=0.02)


def test_multipolygon_touching_the_band_edge_counts_each_pixel_once(live):
    # Each part has one edge on -120, so each zone's piece carries a line from the other part.
    aoi = MultiPolygon([box(-120.02, 38.84, -120.0, 38.86), box(-120.0, 38.86, -119.98, 38.88)])
    assert aoi.is_valid
    nudged = MultiPolygon([box(-120.02, 38.84, -120.0 - 1e-9, 38.86), box(-120.0 + 1e-9, 38.86, -119.98, 38.88)])
    plan = plan_aoi(live, aoi)
    ref = plan_aoi(live, nudged)
    assert plan.zone_names == ["10N", "11N"]
    assert zone_counts(plan) == zone_counts(ref)
    assert_centres_inside(plan, aoi)
    # No 11N pixel centre is west of -120 (those are 10N's), and no 10N centre is east of it.
    lons_11, _ = mask_centres_lonlat(plan.zones[1])
    lons_10, _ = mask_centres_lonlat(plan.zones[0])
    assert (lons_11 >= -120.0).all() and (lons_10 < -120.0).all()


def test_valid_aoi_with_an_edge_on_the_equator_counts_each_pixel_once(live):
    # L shape across the equator at lon 10: the edge (10.01, 0)-(10.02, 0) lies on the equator.
    coords = [(10.0, -0.01), (10.01, -0.01), (10.01, 0.0), (10.02, 0.0), (10.02, 0.01), (10.0, 0.01)]
    aoi = Polygon(coords)
    assert aoi.is_valid
    assert aoi.intersection(zone_band("32S")).geom_type == "GeometryCollection"
    # Nudge only the two vertices of that edge north, off the equator.
    nudged = Polygon([(x, 1e-9) if (y == 0.0 and x > 10.0) else (x, y) for x, y in coords])
    plan = plan_aoi(live, aoi)
    ref = plan_aoi(live, nudged)
    assert plan.zone_names == ["32N", "32S"]
    assert zone_counts(plan) == zone_counts(ref)
    assert_centres_inside(plan, aoi)
    # The N window ends at the N group's bottom edge (northing 0) and the S window starts at the
    # first S row south of the equator (row 1472, from northing 10,000,000).
    n, s = plan.zones
    assert n.rows.stop == n.grid.height
    assert s.rows.start == 1472


def test_invalid_multipart_aoi_on_the_equator_counts_its_parts_once(live):
    # Two boxes sharing an edge (invalid as a MultiPolygon), one of them resting on the equator:
    # make_valid merges them, and the S piece of the result carries the equator edge as a line.
    north = box(10.0, 0.0, 10.01, 0.01)
    south = box(10.01, -0.01, 10.02, 0.005)
    aoi = MultiPolygon([north, south])
    assert not aoi.is_valid
    plan = plan_aoi(live, aoi)
    assert plan.zone_names == ["32N", "32S"]
    parts = [plan_aoi(live, part) for part in (north, south)]
    assert plan.n_pixels == sum(p.n_pixels for p in parts)
    assert_centres_inside(plan, shapely.make_valid(aoi))


def test_make_valid_spike_adds_no_pixels(live):
    square = box(-120.6, 38.8, -120.5, 38.9)
    # A zero-width spike north from the top edge: make_valid keeps it as a LineString.
    spiked = Polygon(
        [
            (-120.6, 38.8),
            (-120.5, 38.8),
            (-120.5, 38.9),
            (-120.55, 38.9),
            (-120.55, 39.0),
            (-120.55, 38.9),
            (-120.6, 38.9),
        ]
    )
    assert not spiked.is_valid
    assert {g.geom_type for g in shapely.get_parts(shapely.make_valid(spiked))} == {"Polygon", "LineString"}
    plan = plan_aoi(live, spiked)
    ref = plan_aoi(live, square)
    assert plan.geometry.geom_type == "Polygon"
    assert plan.geometry.equals(square)
    (p,), (q,) = plan.zones, ref.zones
    assert (p.rows, p.cols) == (q.rows, q.cols)
    np.testing.assert_array_equal(p.mask, q.mask)


def test_make_valid_spike_does_not_count_toward_the_aoi_bounds(live):
    # The spike reaches 84.3 N, past the store's north limit, but the AOI itself ends at 83.95 N.
    spiked = Polygon(
        [
            (-123.05, 83.9),
            (-122.95, 83.9),
            (-122.95, 83.95),
            (-123.0, 83.95),
            (-123.0, 84.3),
            (-123.0, 83.95),
            (-123.05, 83.95),
        ]
    )
    assert not spiked.is_valid
    plan = plan_aoi(live, spiked)
    assert plan.n_pixels == plan_aoi(live, box(-123.05, 83.9, -122.95, 83.95)).n_pixels
    assert plan.geometry.bounds[3] == pytest.approx(83.95)


# ---- G2, G6: points on the equator against live-shaped N and S groups ------------------------

EQ_LON = 10.0
EQ_YEAR = 2020
#: Written region of each group (everything else is fill): the N group's bottom rows and the S
#: group's rows around the equator (rows 1440 to 1471 lie north of it, 1472 on are south).
EQ_ROWS = {"32N": (933_824, 933_888), "32S": (1_440, 1_504)}
EQ_COLS = (44_704, 44_800)
EQ_MARKER = {"32N": 100, "32S": -100}
EQ_CHUNK = 32


def _add_live_zone(root: zarr.Group, zone: str, registration: str) -> None:
    """A zone group with the live dimensions and attributes, holding data only in its EQ window in 2020."""
    _, height = LIVE[zone[-1]]
    g = root.create_group(zone)
    g.attrs.update({**live_attrs(zone, registration), "years_complete": [EQ_YEAR]})
    n = len(YEARS)
    t = YEARS.index(EQ_YEAR)
    r0, r1 = EQ_ROWS[zone]
    c0, c1 = EQ_COLS
    rr, cc = np.meshgrid(np.arange(r0, r1), np.arange(c0, c1), indexing="ij")
    c = EQ_CHUNK
    emb = g.create_array(
        "embeddings", shape=(n, height, LIVE_WIDTH, N_BANDS), dtype="int8", fill_value=0, chunks=(1, c, c, N_BANDS)
    )
    block = np.full((r1 - r0, c1 - c0, N_BANDS), 7, dtype=np.int8)
    block[..., 0] = EQ_MARKER[zone]
    block[..., 1] = rr % 100
    block[..., 2] = cc % 100
    emb[t, r0:r1, c0:c1, :] = block
    scales = g.create_array(
        "scales", shape=(n, height, LIVE_WIDTH), dtype="float32", fill_value=np.nan, chunks=(1, c, c)
    )
    scales[t, r0:r1, c0:c1] = np.full((r1 - r0, c1 - c0), 0.01, dtype=np.float32)
    for name, value in (("s2_obs_count", 50), ("s1_asc_obs_count", 30), ("s1_desc_obs_count", 0)):
        arr = g.create_array(name, shape=(n, height, LIVE_WIDTH), dtype="uint16", fill_value=0, chunks=(1, c, c))
        arr[t, r0:r1, c0:c1] = np.full((r1 - r0, c1 - c0), value, dtype=np.uint16)


@pytest.fixture(scope="module", params=["corner", "centre"])
def equator_reader(request) -> TesseraReader:
    """32N and 32S with live dimensions; ``centre`` stores the pre-fix pixel-centre transform."""
    mem = MemoryStore()
    root = zarr.open_group(mem, mode="w")
    for zone in ("32N", "32S"):
        _add_live_zone(root, zone, request.param)
    return TesseraReader.from_store(mem)


def _expected(zone: str, row: int, col: int) -> np.ndarray:
    return np.array([EQ_MARKER[zone], row % 100, col % 100], dtype=np.float32) * np.float32(0.01)


def test_equator_store_has_the_live_grids(equator_reader):
    n, s = equator_reader.grid("32N"), equator_reader.grid("32S")
    assert (n.left, n.top, n.height, n.width) == (LIVE_LEFT, 9_338_880.0, 933_888, LIVE_WIDTH)
    assert (s.left, s.top, s.height, s.width) == (LIVE_LEFT, 10_014_720.0, 890_880, LIVE_WIDTH)
    # The N group's bottom edge is exactly northing 0, so y == 0 falls one row past it.
    assert n.top - n.height * n.pixel == 0.0
    x, _ = n.from_lonlat().transform(EQ_LON, 0.0)
    assert n.rowcol(x, 0.0)[0] == n.height
    assert s.rowcol(x, 10_000_000.0)[0] == 1472


def test_points_on_the_equator_are_read_from_the_south_group(equator_reader):
    lats = [0.0, -0.0, 1e-9, -1e-9]
    pts = read_points(equator_reader, [EQ_LON] * 4, lats, EQ_YEAR)
    assert pts.zones == ["32S", "32S", "32N", "32S"]
    assert pts.valid.all()
    x, _ = equator_reader.grid("32N").from_lonlat().transform(EQ_LON, 0.0)
    col = int((x - LIVE_LEFT) // 10)
    # lat 0 and -0 land in S row 1472, the pixel just south of the equator, like lat -1e-9.
    np.testing.assert_array_equal(pts.embeddings[0, :3], _expected("32S", 1472, col))
    np.testing.assert_array_equal(pts.embeddings[1, :3], _expected("32S", 1472, col))
    np.testing.assert_array_equal(pts.embeddings[3], pts.embeddings[0])
    # lat 1e-9 is the N group's bottom row.
    np.testing.assert_array_equal(pts.embeddings[2, :3], _expected("32N", 933_887, col))
    assert pts.radar_present is not None and pts.radar_present.all()


@pytest.mark.parametrize("zone", ["32N", "32S"])
def test_points_near_the_equator_hit_their_pixel(equator_reader, zone):
    # Points a quarter pixel in from each corner, so a half-pixel origin error moves them to a neighbour.
    grid = equator_reader.grid(zone)
    rows = [933_884, 933_887] if zone == "32N" else [1_472, 1_475, 1_503]
    cols = [44_704, 44_744, 44_799]
    r, c, fy, fx = (a.ravel() for a in np.meshgrid(rows, cols, [0.25, 0.75], [0.25, 0.75], indexing="ij"))
    xs = LIVE_LEFT + (c + fx) * 10.0
    ys = LIVE[zone[-1]][0] - (r + fy) * 10.0
    lons, lats = grid.to_lonlat().transform(xs, ys)
    pts = read_points(equator_reader, lons, lats, EQ_YEAR, with_radar=False)
    assert set(pts.zones) == {zone}
    assert pts.valid.all()
    expect = np.stack([_expected(zone, int(ri), int(ci)) for ri, ci in zip(r, c, strict=True)])
    np.testing.assert_array_equal(pts.embeddings[:, :3], expect)


# ---- G3: pixel-centre transforms --------------------------------------------------------------


@pytest.mark.parametrize("hemi", ["N", "S"])
def test_centre_registered_transform_uses_the_bbox_corner(hemi):
    top, height = LIVE[hemi]
    zone = f"10{hemi}"
    corner = ZoneGrid.from_attrs(zone, live_attrs(zone, "corner"), (height, LIVE_WIDTH))
    centre = ZoneGrid.from_attrs(zone, live_attrs(zone, "centre"), (height, LIVE_WIDTH))
    assert (corner.left, corner.top) == (LIVE_LEFT, top)
    assert centre == corner
    # A point in the upper-left quarter of pixel (row, col) stays in that pixel.
    row, col = 503_387, 54_116
    assert centre.rowcol(LIVE_LEFT + (col + 0.3) * 10, top - (row + 0.3) * 10) == (row, col)
    assert centre.window_transform(row, col).c == LIVE_LEFT + col * 10


def test_transform_without_a_bbox_is_taken_as_the_corner():
    attrs = {"proj:code": "EPSG:32610", "spatial:transform": [10.0, 0.0, 163845.0, 0.0, -10.0, 9338875.0]}
    g = ZoneGrid.from_attrs("10N", attrs, (933888, 67584))
    assert (g.left, g.top) == (163845.0, 9338875.0)


@pytest.mark.parametrize(
    ("dx", "dy"),
    [
        (10.0, -10.0),  # a whole pixel inside
        (5.0, 0.0),  # half a pixel in x only
        (0.0, -5.0),  # half a pixel in y only
        (-5.0, 5.0),  # half a pixel outside the corner
        (5.0, 5.0),
        (-5.0, -5.0),
        (2.5, -2.5),  # a quarter pixel
        (15.0, -15.0),
        (5.0, -10.0),
    ],
)
def test_transform_off_the_bbox_corner_by_anything_else_is_refused(dx, dy):
    attrs = live_attrs("10N")
    attrs["spatial:transform"] = [10.0, 0.0, LIVE_LEFT + dx, 0.0, -10.0, 9_338_880.0 + dy]
    with pytest.raises(TesseraError, match="disagrees with spatial:bbox"):
        ZoneGrid.from_attrs("10N", attrs, (933_888, LIVE_WIDTH))


# ---- G4: coordinates out of range -----------------------------------------------------------


@pytest.mark.parametrize(("lon", "lat"), [(190.0, 10.0), (-190.0, 10.0), (180.5, 60.0), (10.0, 95.0), (10.0, -95.0)])
def test_out_of_range_coordinates_raise(lon, lat):
    with pytest.raises(ValueError, match="outside"):
        zone_for(lon, lat)


def test_zone_number_range():
    assert zone_number(-180.0) == 1
    assert zone_number(180.0) == 60
    assert zone_for(180.0, -10.0) == "60S"
    assert zone_for(10.0, 90.0) == "32N"
    assert zone_for(10.0, -90.0) == "32S"
    for bad in (190.0, -180.000001, 360.0, float("inf"), float("nan")):
        with pytest.raises(ValueError, match="-180 to 180"):
            zone_number(bad)


@pytest.mark.parametrize(("lons", "lats"), [([-170.0, 190.0], [10.0, 10.0]), ([10.0, 10.0], [10.0, 95.0])])
def test_read_points_refuses_out_of_range_coordinates_before_reading(live, lons, lats):
    # The stub fails on any chunk read, so the error must come first.
    with pytest.raises(ValueError, match="outside"):
        read_points(live, lons, lats, 2020)


# ---- TQ14: window_for_bounds snaps outward on every side --------------------------------------


@pytest.mark.parametrize(
    ("bounds", "rows", "cols"),
    [
        ((3.0, 903.0, 97.0, 953.0), (4, 10), (0, 10)),
        ((13.0, 917.0, 47.0, 961.0), (3, 9), (1, 5)),
        ((10.0, 900.0, 50.0, 960.0), (4, 10), (1, 5)),  # on pixel edges: no growth
        ((19.9, 950.1, 20.1, 959.9), (4, 5), (1, 3)),
        ((-50.0, -100.0, 2000.0, 1100.0), (0, 100), (0, 100)),  # clipped to the grid
    ],
)
def test_window_for_bounds_snaps_outward(bounds, rows, cols):
    g = ZoneGrid(zone="10N", epsg=32610, left=0.0, top=1000.0, height=100, width=100)
    assert g.window_for_bounds(*bounds) == (slice(*rows), slice(*cols))


def test_window_for_bounds_on_the_live_grid():
    g = live_grid("10N")
    rows, cols = g.window_for_bounds(700_003.0, 4_200_003.0, 700_097.0, 4_200_047.0)
    assert (rows, cols) == (slice(513_883, 513_888), slice(53_616, 53_626))


def test_plan_keeps_edge_pixels_whose_centres_are_inside(live):
    # A projected box whose four edges each lie 2 m beyond a row or column of pixel centres. Every
    # edge pixel has its centre inside the AOI, and only outward snapping keeps all of them.
    g = live_grid("10N")
    r0, r1, c0, c1 = 513_800, 513_841, 53_600, 53_633
    x0, x1 = g.left + (c0 + 0.5) * 10 - 2, g.left + (c1 - 0.5) * 10 + 2
    y0, y1 = g.top - (r1 - 0.5) * 10 - 2, g.top - (r0 + 0.5) * 10 + 2
    projected = shapely.segmentize(box(x0, y0, x1, y1), 20.0)
    aoi = shapely.transform(projected, g.to_lonlat().transform, interleaved=False)
    (p,) = plan_aoi(live, aoi).zones
    assert (p.rows, p.cols) == (slice(r0, r1), slice(c0, c1))
    assert p.n_pixels == (r1 - r0) * (c1 - c0)
    assert p.mask.all()


# ---- TQ15: the zone limit at its real value -------------------------------------------------


def test_four_zone_equator_corner_is_planned_at_the_default_limit(live):
    aoi = box(-120.01, -0.01, -119.99, 0.01)
    plan = plan_aoi(live, aoi)
    assert plan.zone_names == ["10N", "10S", "11N", "11S"]
    assert all(p.n_pixels > 0 for p in plan.zones)
    assert_centres_inside(plan, aoi)
    area_m2 = abs(Geod(ellps="WGS84").geometry_area_perimeter(aoi)[0])
    assert plan.n_pixels * 100 == pytest.approx(area_m2, rel=0.02)


@pytest.mark.parametrize(
    ("aoi", "n"),
    [
        (box(-120.01, -0.01, -113.99, 0.01), 6),
        # The corner box plus an arm north of the equator into zone 12: 10N, 10S, 11N, 11S, 12N.
        (shapely.union_all([box(-120.01, -0.01, -119.99, 0.01), box(-119.99, 0.001, -113.99, 0.01)]), 5),
    ],
)
def test_more_than_four_zones_is_refused_before_any_grid_read(live, aoi, n):
    with pytest.raises(AOIError, match=f"touches {n} zone groups .*the limit is 4"):
        plan_aoi(live, aoi)
    assert live.grid_calls == []


# ---- cost-perf F1: no continent-sized masks ----------------------------------------------------


@pytest.fixture
def no_rasterize(monkeypatch):
    """Fail the test if plan_aoi gets as far as rasterizing a mask."""

    def refuse(*args, **kwargs):
        raise AssertionError("plan_aoi rasterized a mask for an AOI it should refuse")

    monkeypatch.setattr(tio, "rasterize", refuse)


def _plan_peak(reader: Any, aoi: BaseGeometry, **kwargs: Any) -> tuple[AOIPlan | AOIError, int]:
    """Plan an AOI and return the plan (or the AOIError) with the peak traced allocation in bytes.

    numpy reports its buffers to tracemalloc, so a mask shows up here as soon as it is allocated.
    """
    started = not tracemalloc.is_tracing()
    if started:
        tracemalloc.start()
    tracemalloc.reset_peak()
    base, _ = tracemalloc.get_traced_memory()
    try:
        result: AOIPlan | AOIError = plan_aoi(reader, aoi, **kwargs)
    except AOIError as exc:
        result = exc
    finally:
        _, peak = tracemalloc.get_traced_memory()
        if started:
            tracemalloc.stop()
    return result, peak - base


def test_sparse_far_apart_parts_are_refused_before_any_mask(live, no_rasterize):
    # Two 1 km2 parcels 1,700 km apart in 10N: a 166,629 x 42,006 pixel window (6.5 GiB of mask).
    aoi = MultiPolygon([box(-125.5, 33.0, -125.49, 33.009), box(-120.51, 48.0, -120.5, 48.009)])
    result, peak = _plan_peak(live, aoi)
    assert isinstance(result, AOIError)
    assert "far apart" in str(result)
    assert peak < 16 * 2**20
    assert live.grid_calls == ["10N"]


def test_sparse_part_in_a_later_zone_is_refused_before_any_mask(live, no_rasterize):
    # A compact box in 10N plus two parcels 1,700 km apart in 11N.
    aoi = MultiPolygon(
        [box(-120.6, 38.8, -120.55, 38.85), box(-119.5, 33.0, -119.49, 33.009), box(-115.0, 48.0, -114.99, 48.009)]
    )
    with pytest.raises(AOIError, match="AOI part in 11N"):
        plan_aoi(live, aoi)


def test_over_area_limit_is_refused_before_any_grid_or_mask(live, no_rasterize):
    aoi = box(-121.0, 38.0, -119.5, 39.0)  # about 14,500 km2 over 10N and 11N
    assert MAX_AOI_KM2 == 10_000.0
    result, peak = _plan_peak(live, aoi)
    assert isinstance(result, AOIError)
    assert "over the 10,000 km2 limit" in str(result)
    assert peak < 16 * 2**20
    assert live.grid_calls == []


def test_area_limit_is_in_km2_and_can_be_lifted(live):
    aoi = box(-120.6, 38.8, -120.59, 38.81)  # about 0.96 km2
    with pytest.raises(AOIError, match="limit"):
        plan_aoi(live, aoi, max_area_km2=0.9)
    assert plan_aoi(live, aoi, max_area_km2=1.0).n_pixels > 0
    assert plan_aoi(live, aoi, max_area_km2=None).n_pixels > 0


def _utm_to_lonlat(geometry: BaseGeometry, epsg: int = 32610) -> BaseGeometry:
    to_ll = Transformer.from_crs(epsg, 4326, always_xy=True)
    return shapely.transform(shapely.segmentize(geometry, 100.0), to_ll.transform, interleaved=False)


def test_long_thin_diagonal_is_refused_before_any_mask(live, no_rasterize):
    # 100 m wide, 50 km long, at 45 degrees: about 50,000 AOI pixels in a 12 million pixel window.
    corridor = LineString([(600_000, 4_300_000), (635_000, 4_335_000)]).buffer(50.0, cap_style="flat")
    result, peak = _plan_peak(live, _utm_to_lonlat(corridor))
    assert isinstance(result, AOIError)
    assert "long and thin" in str(result)
    assert peak < 16 * 2**20


def _ring(width_m: float, side_m: float = 30_000.0) -> BaseGeometry:
    x0, y0 = 600_000.0, 4_300_000.0
    outer = box(x0, y0, x0 + side_m, y0 + side_m)
    inner = box(x0 + width_m, y0 + width_m, x0 + side_m - width_m, y0 + side_m - width_m)
    return _utm_to_lonlat(outer.difference(inner))


def test_window_to_aoi_ratio_limit(live):
    # A 30 km square ring 1.5 km wide fills about 1/5 of its window: accepted. At 0.5 km wide it
    # fills about 1/15: refused, even though the window is only 9 million pixels.
    plan = plan_aoi(live, _ring(1_500.0))
    (p,) = plan.zones
    ratio = p.shape[0] * p.shape[1] / p.n_pixels
    assert 4.5 < ratio < 6.0
    with pytest.raises(AOIError, match="far apart, or it is long and thin"):
        plan_aoi(live, _ring(500.0))


def test_small_windows_are_allowed_however_sparse(live):
    # Two 100 m parcels 10 km apart: a window over 4,000 times the AOI but under 4 million pixels.
    aoi = MultiPolygon([box(-120.6, 38.8, -120.599, 38.801), box(-120.5, 38.88, -120.499, 38.881)])
    (p,) = plan_aoi(live, aoi).zones
    assert p.shape[0] * p.shape[1] < 4_000_000
    assert p.shape[0] * p.shape[1] > 4_000 * p.n_pixels


def test_plan_mask_costs_one_byte_per_window_pixel(live):
    # The mask is rasterized into a preallocated uint8 and viewed as bool, with no full-size copy.
    result, peak = _plan_peak(live, box(-120.6, 38.8, -120.5, 38.89))
    assert isinstance(result, AOIPlan)
    (p,) = result.zones
    window_px = p.shape[0] * p.shape[1]
    assert window_px > 500_000
    assert p.mask.dtype == np.bool_
    assert p.mask.shape == p.shape
    assert peak < 1.5 * window_px
