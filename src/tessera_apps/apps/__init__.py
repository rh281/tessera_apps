"""The four climate applications and the plumbing they share.

Each app is a function (``run_wildfire``, ``run_deforest``, ``run_wetland`` and, from a later
checkpoint, ``run_restore``) with a matching ``tessera-apps`` command. They share
:mod:`tessera_apps.apps._common`: AOI loading, the run preparation that checks every year and
skips a run whose outputs are current, the zone mosaic that cleans masks across UTM seams, and the
report written last.
"""

from __future__ import annotations

from tessera_apps.apps._common import (
    AppInputError,
    LoadedAOI,
    RunContext,
    StrictValidationError,
    finish_run,
    load_aoi,
    mosaic_and_clean,
    prepare_run,
)
from tessera_apps.apps.deforest import DeforestResult, run_deforest
from tessera_apps.apps.wetland import WetlandResult, run_wetland
from tessera_apps.apps.wildfire import WildfireResult, run_wildfire

__all__ = [
    "AppInputError",
    "DeforestResult",
    "LoadedAOI",
    "RunContext",
    "StrictValidationError",
    "WetlandResult",
    "WildfireResult",
    "finish_run",
    "load_aoi",
    "mosaic_and_clean",
    "prepare_run",
    "run_deforest",
    "run_wetland",
    "run_wildfire",
]
