"""The hosted demo: the bundled demos' precomputed outputs, and a map click that runs the wildfire recovery trajectory.

Run it from the repository root with ``streamlit run demo/app.py``. The page has two parts:

- **A map click.** The map opens on the Caldor demo AOI with its outline drawn. A click runs the
  wildfire recovery trajectory of the clicked pixel (``demo/trajectory.py``): it reads the one
  2.56 km chunk that holds the pixel, for the years the curve needs, from the anonymous store at a
  pinned snapshot, and draws the pixel's similarity to its pre-fire year in every year and its
  recovery index. Inside the Caldor AOI at the demo's years it takes the precomputed Caldor run's
  threshold and unburned reference; elsewhere it applies the app's rules to the chunk alone. It
  shows the chunk payload bytes read and the wall time of every click; a click in a chunk and
  years already read reads nothing and says so.
- **The bundled demos.** A dropdown of the five demos shows each one's preview, summary table,
  headline numbers with what each is scored against, labels, warnings and the command that
  reproduces it, all from ``demo/precomputed`` (made by ``demo/precompute.py``). Choosing a demo
  reads nothing from the store.

No AWS credentials are needed or used. Every failure the page can meet is shown with ``st.error``
and a message that says what to change, never a traceback.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

import altair as alt
import folium
import pandas as pd
import streamlit as st
from shapely.geometry import mapping
from streamlit_folium import st_folium

from tessera_apps import __version__
from tessera_apps.demo import demo_aoi
from tessera_apps.io import FIRST_YEAR, LAST_YEAR

DEMO_DIR = Path(__file__).resolve().parent
# streamlit puts demo/ on sys.path; the repository root goes there too, so the page imports its
# click logic as demo.trajectory, the name the tests and mypy use.
if str(DEMO_DIR.parent) not in sys.path:
    sys.path.insert(0, str(DEMO_DIR.parent))

from demo import trajectory  # noqa: E402
from demo.trajectory import ClickError, DemoStore, Trajectory  # noqa: E402

PRECOMPUTED = DEMO_DIR / "precomputed"
INDEX = PRECOMPUTED / "index.json"
#: The two series colours (categorical slots 1 and 2 of the dataviz reference palette), by theme.
SERIES_COLOURS = {"light": ("#2a78d6", "#eb6834"), "dark": ("#3987e5", "#d95926")}
#: The numeric columns of the click's table.
NUMBER_COLUMNS = ("similarity", "reference median", "R_t", "window share recovered", "window median R_t")
#: What the page's byte counts leave out, measured on 2026-10-26 (loopback bytes against the
#: reader's count, through the build machine's proxy).
BYTES_NOTE = (
    "Bytes are chunk payload bytes, counted as in the app's reports. The store's manifests and metadata are not "
    "counted: on 2026-10-26 they were about 2.2 MB when the store opened and about 2.3 MB more on the first click."
)
#: How the dropdown names each app.
APP_NAMES = {
    "wildfire": "wildfire: burn scar and recovery",
    "deforest": "deforest: forest converted",
    "wetland": "wetland: mangrove extent",
    "restore": "restore: site against a matched control",
}

st.set_page_config(page_title="tessera_apps demo", page_icon=":fire:", layout="wide")

# Every click's bytes and wall time also go to the server log (streamlit reruns this script, so the
# handler is added once).
_CLICK_LOG = logging.getLogger(trajectory.__name__)
if not _CLICK_LOG.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s"))
    _CLICK_LOG.addHandler(_handler)
    _CLICK_LOG.setLevel(logging.INFO)


# --------------------------------------------------------------------------------------------
# Cached resources
# --------------------------------------------------------------------------------------------


@st.cache_resource(show_spinner=False)
def shared_store() -> DemoStore:
    """The reader, snapshot and window cache every session shares. Opening reads store metadata only."""
    return trajectory.open_store()


@st.cache_data(show_spinner=False)
def load_index(path: str) -> dict[str, Any]:
    """The precomputed demos' index (``demo/precomputed/index.json``)."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def caldor_outline() -> dict[str, Any]:
    """The Caldor demo AOI as GeoJSON, with its lon/lat bounds."""
    geometry = demo_aoi("caldor")
    return {"geojson": mapping(geometry), "bounds": list(geometry.bounds)}


def _theme() -> str:
    try:
        kind = st.context.theme.type
    except AttributeError:
        kind = None
    return "dark" if kind == "dark" else "light"


# --------------------------------------------------------------------------------------------
# The map click
# --------------------------------------------------------------------------------------------


def _years_inputs() -> tuple[int, int, int]:
    first, last = FIRST_YEAR, LAST_YEAR
    c1, c2, c3 = st.columns(3)
    year_a = c1.number_input(
        "Pre-fire year", min_value=first, max_value=last, value=trajectory.DEFAULT_YEAR_A, step=1, key="year_a"
    )
    year_b = c2.number_input(
        "Post-fire year", min_value=first, max_value=last, value=trajectory.DEFAULT_YEAR_B, step=1, key="year_b"
    )
    through = c3.number_input("Last recovery year", min_value=first, max_value=last, value=last, step=1, key="through")
    return int(year_a), int(year_b), int(through)


def _map(result: Trajectory | None) -> dict[str, Any] | None:
    """Draw the map on the Caldor AOI, with the last click's window, and return what streamlit-folium reports."""
    outline = caldor_outline()
    west, south, east, north = outline["bounds"]
    m = folium.Map(
        location=[(south + north) / 2, (west + east) / 2],
        zoom_start=10,
        tiles=None,
        control_scale=True,
        world_copy_jump=True,  # panning across the antimeridian jumps back to the main copy of the world
    )
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Tiles: Esri. Source: Esri, Maxar, Earthstar Geographics, and the GIS User Community",
        name="Imagery",
    ).add_to(m)
    folium.TileLayer("OpenStreetMap", name="OpenStreetMap").add_to(m)
    folium.GeoJson(
        outline["geojson"],
        name="Caldor demo AOI",
        style_function=lambda _: {"color": "#eb6834", "weight": 2, "fill": False},
        tooltip="Caldor demo AOI (provisional: derived from Hansen 2021 loss)",
    ).add_to(m)
    m.fit_bounds([[south, west], [north, east]])
    folium.LayerControl(collapsed=True).add_to(m)
    clicked = folium.FeatureGroup(name="Last click")
    if result is not None:
        folium.GeoJson(
            mapping(result.window_lonlat),
            style_function=lambda _: {"color": "#2a78d6", "weight": 2, "fill": False},
            tooltip=f"The chunk read: zone {result.zone}, chunk {result.chunk}",
        ).add_to(clicked)
        folium.CircleMarker(
            [result.lat, result.lon], radius=5, color="#ffffff", weight=2, fill=True, fill_color="#2a78d6"
        ).add_to(clicked)
    return st_folium(
        m,
        key="map",
        height=520,
        use_container_width=True,
        returned_objects=["last_clicked"],
        feature_group_to_add=clicked,
        wrap_longitude=True,  # a click on another copy of the world reports its longitude in -180 to 180
    )


def _run(lon: float, lat: float, year_a: int, year_b: int, through: int) -> dict[str, Any]:
    """Run one click; the answer or the message to show, never an exception."""
    opened: float | None = None
    try:
        with st.spinner("Reading the chunk that holds the pixel..."):
            store = shared_store()
            opened = None if store.clicks else store.open_s
            found = trajectory.run_click(store, lon, lat, year_a=year_a, year_b=year_b, through=through)
            return {"trajectory": found, "open_s": opened}
    except ClickError as exc:
        return {"error": str(exc), "bytes_read": exc.bytes_read, "wall_s": exc.wall_s, "open_s": opened}
    except Exception as exc:  # the page shows every failure as a message, never a traceback
        return {
            "error": f"The click failed unexpectedly ({type(exc).__name__}: {exc}). Click again; if it fails again, "
            "the store may be unreachable from this host: it needs anonymous HTTPS access to S3 in us-west-2.",
            "open_s": opened,
        }


def _open_line(state: dict[str, Any]) -> str | None:
    opened = state.get("open_s")
    if opened is None:
        return None
    return f"This first click also opened the store, which reads its metadata and no chunk, in {opened:.2f} s."


def _cost_line(t: Trajectory) -> str:
    fetched = _years_text(t.years_fetched)
    cached = _years_text(t.years_cached)
    where = f"zone {t.zone} chunk {t.chunk} ({t.window_km[0]:g} km by {t.window_km[1]:g} km)"
    if t.from_cache:
        return (
            f"Read 0 chunk payload bytes from the store: {cached} of {where} came from the cache (keyed by snapshot, "
            f"zone, chunk and year). The click took {t.wall_s:.2f} s. Snapshot {t.snapshot}."
        )
    text = (
        f"Read {t.bytes_read:,} chunk payload bytes ({t.bytes_read / 1e6:.1f} MB) of {where} for {fetched} from "
        f"the store in {t.read_s:.2f} s"
    )
    if t.years_cached:
        text += f"; {cached} came from the cache"
    return f"{text}. The click took {t.wall_s:.2f} s in all. Snapshot {t.snapshot}."


def _years_text(years: list[int]) -> str:
    if not years:
        return "no year"
    if years == list(range(years[0], years[-1] + 1)) and len(years) > 2:
        return f"{years[0]} to {years[-1]}"
    return ", ".join(str(y) for y in years)


def _pixel_line(t: Trajectory) -> str:
    parts = [f"Pixel ({t.lon:.5f}, {t.lat:.5f}): zone {t.zone}, row {t.row:,}, column {t.col:,}."]
    if t.threshold is not None and t.run is not None:
        method = "Otsu" if t.run["method"] == "otsu" else t.run["method"]
        parts.append(
            f"Threshold and unburned reference of {t.run['label']}: threshold {t.threshold['threshold']:.3f} "
            f"({method} over its whole AOI), and that run's unburned reference ({t.reference_pixels:,} pixels in "
            f"{t.year_b}). {t.burned_pixels:,} of the window's {t.window_pixels:,} pixels burned by it."
        )
    elif t.threshold is not None:
        parts.append(
            f"Window threshold {t.threshold['threshold']:.3f} ({t.threshold['method']}): {t.burned_pixels:,} of "
            f"{t.window_pixels:,} pixels burned, {t.reference_pixels:,} in the window's unburned reference."
        )
    if t.pixel_distance is not None:
        state = {True: "burned", False: "did not burn", None: "is not judged"}[t.pixel_burned]
        parts.append(f"This pixel {state} ({t.year_a} to {t.year_b} distance {t.pixel_distance:.3f}).")
    return " ".join(parts)


def _reference_series(t: Trajectory) -> str:
    """The chart's name for the unburned reference the click used."""
    if t.run is None:
        return "Window's unburned reference, median"
    return f"{t.run['name'].capitalize()} run's unburned reference, median"


def _frames(t: Trajectory) -> tuple[pd.DataFrame, pd.DataFrame]:
    reference = _reference_series(t)
    sim_rows: list[dict[str, Any]] = []
    r_rows: list[dict[str, Any]] = []
    for p in t.points:
        regime = "no data" if p.radar is None else ("radar" if p.radar else "optical only")
        if p.similarity is not None:
            sim_rows.append(
                {
                    "year": p.year,
                    "value": p.similarity,
                    "series": "This pixel",
                    "marker": "same radar regime as the pre-fire year" if p.comparable else "radar regime differs",
                    "regime": regime,
                    "role": p.role,
                }
            )
        if p.reference_median is not None:
            sim_rows.append(
                {
                    "year": p.year,
                    "value": p.reference_median,
                    "series": reference,
                    "marker": "same radar regime as the pre-fire year",
                    "regime": "",
                    "role": p.role,
                }
            )
        if p.recovery_index is not None:
            r_rows.append({"year": p.year, "value": p.recovery_index, "series": "This pixel", "role": p.role})
        if p.window_median_recovery_index is not None:
            r_rows.append(
                {
                    "year": p.year,
                    "value": p.window_median_recovery_index,
                    "series": "Window's burned pixels, median",
                    "role": p.role,
                }
            )
    return pd.DataFrame(sim_rows), pd.DataFrame(r_rows)


def _chart(frame: pd.DataFrame, *, title: str, y_title: str, series: list[str], rules: list[float]) -> Any:  # noqa: ANN401
    colours = SERIES_COLOURS[_theme()]
    colour = alt.Color(
        "series:N",
        scale=alt.Scale(domain=series, range=list(colours[: len(series)])),
        title=None,
        legend=alt.Legend(orient="top", direction="horizontal", labelLimit=400),
    )
    lo = min(float(frame["value"].min()), *rules) - 0.05
    hi = max(float(frame["value"].max()), *rules) + 0.05
    x = alt.X("year:O", title=None, axis=alt.Axis(labelAngle=0))
    y = alt.Y("value:Q", title=y_title, scale=alt.Scale(domain=[lo, hi]))
    tooltip = [alt.Tooltip("year:O"), alt.Tooltip("series:N"), alt.Tooltip("value:Q", format=".3f")]
    base = alt.Chart(frame).encode(x=x, y=y, color=colour)
    if "marker" in frame:
        tooltip.append(alt.Tooltip("regime:N", title="radar"))
        same = frame["marker"] == "same radar regime as the pre-fire year"
        line = alt.Chart(frame[same]).mark_line(strokeWidth=2).encode(x=x, y=y, color=colour)
        filled = (
            alt.Chart(frame[same])
            .mark_point(size=70, filled=True, opacity=1)
            .encode(x=x, y=y, color=colour, tooltip=tooltip)
        )
        hollow = (
            alt.Chart(frame[~same])
            .mark_point(size=70, filled=False, strokeWidth=2, opacity=1)
            .encode(x=x, y=y, color=colour, tooltip=tooltip)
        )
        layers = [line, filled, hollow]
    else:
        layers = [
            base.mark_line(strokeWidth=2),
            base.mark_point(size=70, filled=True, opacity=1).encode(tooltip=tooltip),
        ]
    rule_frame = pd.DataFrame({"value": rules})
    rule = alt.Chart(rule_frame).mark_rule(strokeWidth=1, color="#898781").encode(y="value:Q")
    return alt.layer(rule, *layers).properties(title=title, height=260)


def _show_result(state: dict[str, Any]) -> None:
    if "error" in state:
        st.error(state["error"])
        if "wall_s" in state:
            st.caption(
                f"This click read {state['bytes_read']:,} chunk payload bytes from the store in "
                f"{state['wall_s']:.2f} s."
            )
        if _open_line(state):
            st.caption(_open_line(state))
        return
    t: Trajectory = state["trajectory"]
    (st.info if t.from_cache else st.success)(_cost_line(t))
    st.caption(BYTES_NOTE)
    if _open_line(state):
        st.caption(_open_line(state))
    st.markdown(_pixel_line(t))
    for note in t.notes[:-1]:
        st.warning(note)
    st.info(t.notes[-1])
    sims, rs = _frames(t)
    if not sims.empty:
        st.altair_chart(
            _chart(
                sims,
                title=f"Cosine similarity to the pixel's own {t.year_a} vector",
                y_title="similarity",
                series=["This pixel", _reference_series(t)],
                rules=[1.0],
            ),
            width="stretch",
        )
        st.caption(
            "Hollow markers: the pixel's radar regime differs from the pre-fire year's, so the app leaves the pair out."
        )
    if not rs.empty:
        st.altair_chart(
            _chart(
                rs,
                title="Recovery index R_t = (s_t - s_post) / (s_ref_t - s_post)",
                y_title="recovery index",
                series=["This pixel", "Window's burned pixels, median"],
                rules=[0.0, 1.0],
            ),
            width="stretch",
        )
        whose = "the window's" if t.run is None else f"{t.run['label']}'s"
        st.caption(
            f"0 is the state just after the fire ({t.year_b}); 1 is as similar to the pre-fire vector as {whose} "
            "unburned reference that year."
        )
    st.dataframe(
        _table(t),
        hide_index=True,
        column_config={k: st.column_config.NumberColumn(format="%.4f") for k in NUMBER_COLUMNS},
    )


def _yes(value: bool | None) -> str:
    return "" if value is None else ("yes" if value else "no")


def _table(t: Trajectory) -> pd.DataFrame:
    """The click's curve as the page's table: one row per year."""
    rows = [
        {
            "year": p.year,
            "role": p.role,
            "similarity": p.similarity,
            "radar": "no data" if p.radar is None else _yes(p.radar),
            "same radar regime": _yes(p.radar_matches),
            "reference median": p.reference_median,
            "R_t": p.recovery_index,
            "window share recovered": p.window_recovered_fraction,
            "window median R_t": p.window_median_recovery_index,
        }
        for p in t.points
    ]
    frame = pd.DataFrame(rows)
    for column in NUMBER_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def click_section() -> None:
    """The map, its year inputs, and the last click's trajectory."""
    st.header("Wildfire recovery at one pixel")
    st.markdown(
        "Click the map to run the wildfire app's recovery trajectory for one 10 m pixel. The click reads only the "
        "2.56 km chunk of the store that holds the pixel, for the pre-fire year and every year after it (and the "
        "year before it when the window's own rules need it), and computes the pixel's similarities and recovery "
        "index with the app's own functions. Inside the Caldor outline at the demo's years (pre-fire 2020, "
        "post-fire 2022), it takes the threshold and unburned reference of the precomputed Caldor run, so the "
        "burned verdict and recovery index are that run's. Anywhere else, or at other years, it applies the app's "
        "threshold rule and unburned reference to the 2.56 km window alone, which can disagree with a run over a "
        "whole fire."
    )
    year_a, year_b, through = _years_inputs()
    state = st.session_state.get("click")
    shown = state.get("trajectory") if state else None
    left, right = st.columns([1, 1])
    with left:
        out = _map(shown)
    clicked = (out or {}).get("last_clicked")
    if clicked:
        lon, lat = float(clicked["lng"]), float(clicked["lat"])
        key = (round(lon, 7), round(lat, 7), year_a, year_b, through)
        if state is None or state.get("key") != key:
            state = {"key": key, **_run(lon, lat, year_a, year_b, through)}
            st.session_state["click"] = state
            _log(state, lon, lat, year_a, year_b)
            if "trajectory" in state:
                st.rerun()  # redraw the map with the window just read
    with right:
        if state is None:
            st.info(
                "Click a pixel inside the Caldor outline (or anywhere on land) to read its chunk. The first click "
                "opens the store; nothing is read before it."
            )
        else:
            _show_result(state)
    log = st.session_state.get("log", [])
    if log:
        with st.expander(f"Every click this session ({len(log)})", expanded=False):
            st.dataframe(pd.DataFrame(log), hide_index=True)


def _log(state: dict[str, Any], lon: float, lat: float, year_a: int, year_b: int) -> None:
    t: Trajectory | None = state.get("trajectory")
    row = {
        "lon": round(lon, 5),
        "lat": round(lat, 5),
        "years": f"{year_a} to {year_b}",
        "zone": t.zone if t else None,
        "chunk": str(t.chunk) if t else None,
        "bytes read": t.bytes_read if t else state.get("bytes_read", 0),
        "read s": round(t.read_s, 2) if t else None,
        "wall s": round(t.wall_s if t else state.get("wall_s", 0.0), 2),
        "from cache": t.from_cache if t else None,
        "result": "ok" if t else state["error"][:80],
    }
    st.session_state.setdefault("log", []).append(row)


# --------------------------------------------------------------------------------------------
# The bundled demos
# --------------------------------------------------------------------------------------------


def _md(text: str) -> str:
    """Text for st.markdown, with the characters it would read as markup escaped."""
    for ch in "\\`*_$<>[]":
        text = text.replace(ch, "\\" + ch)
    return text


def _bytes_text(n: int) -> str:
    return f"{n / 1e9:.2f} GB" if n >= 1e9 else f"{n / 1e6:.0f} MB"


def demo_section() -> None:
    """The dropdown of the bundled demos and the chosen demo's precomputed outputs."""
    st.header("The bundled demos")
    try:
        index = load_index(str(INDEX))
        demos = list(index["demos"])
    except (OSError, ValueError, KeyError, TypeError) as exc:
        st.error(
            f"Cannot read {INDEX.relative_to(DEMO_DIR.parent)} ({type(exc).__name__}: {exc}). Make it with "
            "python demo/precompute.py --work DIR, run from the repository root."
        )
        return
    st.markdown(
        f"Each demo was run with the package's own command at snapshot `{index.get('snapshot')}` and its outputs "
        "were copied into `demo/precomputed`, so choosing one reads nothing from the store."
    )
    by_name = {d["name"]: d for d in demos}
    name = st.selectbox(
        "Demo",
        list(by_name),
        format_func=lambda n: f"{n} ({APP_NAMES.get(by_name[n]['app'], by_name[n]['app'])})",
        key="demo",
    )
    _show_demo(by_name[name])


def _show_demo(d: dict[str, Any]) -> None:
    aoi, years = d["aoi"], d["years"]
    st.subheader(aoi.get("title", d["name"]))
    st.caption(
        f"App `{d['app']}`, AOI {aoi.get('area_ha', 0):,.1f} ha in zone {', '.join(aoi.get('zones', []))}, "
        f"years {years.get('year_a')} and {years.get('year_b')} (read: {_years_text(list(years.get('read', [])))})."
    )
    for label in d.get("labels", []):
        st.info(label)
    if years.get("note"):
        st.info(years["note"])
    _headline(d)
    files = d.get("files", {})
    preview = files.get("preview", {}).get("path")
    if preview and (PRECOMPUTED / preview).is_file():
        st.image(str(PRECOMPUTED / preview), caption=f"{preview} (the app's preview PNG)", width="stretch")
    else:
        st.error(f"The preview {preview} is missing from demo/precomputed; rerun python demo/precompute.py.")
    summary = files.get("summary_csv", {}).get("path")
    if summary and (PRECOMPUTED / summary).is_file():
        st.markdown(f"**Summary table** (`{summary}`)")
        st.dataframe(pd.read_csv(PRECOMPUTED / summary), hide_index=True)
    else:
        st.error(f"The summary table {summary} is missing from demo/precomputed; rerun python demo/precompute.py.")
    warnings = d.get("warnings", [])
    if warnings:
        st.markdown(f"**Warnings the run recorded** ({len(warnings)})")
        for w in warnings:
            st.warning(w)
    else:
        st.markdown("**Warnings the run recorded:** none.")
    st.markdown("**The command that reproduces it**")
    st.code(d["command"], language="bash", wrap_lines=True)
    versions = d.get("versions", {})
    st.caption(
        f"Run from {d.get('command_cwd', 'the work directory')}. It read {_bytes_text(int(d['bytes_read']))} "
        f"({int(d['bytes_read']):,} bytes) in {float(d['wall_s']):.0f} s at snapshot {d.get('snapshot')}, with "
        f"tessera_apps {versions.get('tessera_apps')} and geotessera {versions.get('geotessera')}, on "
        f"{d.get('run_created_utc')}."
    )
    for item in d.get("inputs", []):
        st.caption(f"Input {item['file']} (sha256 {item['sha256'][:16]}): {item['made_by']}.")
    validation = files.get("validation", {}).get("path")
    if validation and (PRECOMPUTED / validation).is_file():
        st.download_button(
            f"Download {Path(validation).name}",
            data=(PRECOMPUTED / validation).read_bytes(),
            file_name=Path(validation).name,
            mime="application/json",
            key=f"validation_{d['name']}",
        )


def _headline(d: dict[str, Any]) -> None:
    items = d.get("headline", [])
    plain = [h for h in items if "scored_against" not in h]
    scores = [h for h in items if "scored_against" in h]
    numbers = [h for h in plain if len(str(h["display"])) <= 24]
    for start in range(0, len(numbers), 4):
        cols = st.columns(4)
        for col, h in zip(cols, numbers[start : start + 4], strict=False):
            col.metric(h["label"], h["display"])
    for h in plain:
        if len(str(h["display"])) > 24:
            st.markdown(f"**{h['label']}:** {h['display']}")
    if scores:
        st.markdown("**Scores, and what each is scored against**")
        groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for h in scores:
            groups.setdefault((h["scored_against"], h["reference"]), []).append(h)
        for (against, reference), group in groups.items():
            values = "; ".join(f"{_md(h['label'])} **{_md(h['display'])}**" for h in group)
            st.markdown(f"- {values}  \n  Scored against {_md(against)}.  \n  *{_md(reference)}*")
    if d.get("app_headline"):
        st.caption(f"The app's own headline: {d['app_headline']}")


# --------------------------------------------------------------------------------------------
# The page
# --------------------------------------------------------------------------------------------


def main() -> None:
    """Draw the page."""
    st.title("tessera_apps demo")
    st.markdown(
        f"Climate apps on the TESSERA v1.1 global embeddings, read anonymously from the dClimate store "
        f"(`s3://tessera-embeddings/v1.1/dclimate.icechunk`) with tessera_apps {__version__}. No AWS account or "
        f"credentials are used. The map click reads snapshot `{trajectory.resolve_snapshot()}`."
    )
    click_section()
    st.divider()
    demo_section()


if __name__ == "__main__":
    main()
