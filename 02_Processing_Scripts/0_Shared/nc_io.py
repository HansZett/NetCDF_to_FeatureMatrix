# -*- coding: utf-8 -*-
"""
nc_io.py — Shared NetCDF conventions & I/O helper
=================================================

Single source of truth for how every NetCDF file in this project is written,
so that all datasets (C3S_LCC, ERA5_Land, SoilGrids, HydroSHEDS…) end up
coherent in structure, orientation, metadata and fill-value handling.

The module lives in ``02_Processing_Scripts/0_Shared/`` and is imported by
every dataset script (and by the extraction scripts in
``02_Dataframes_Processing_Scripts``) as follows:

    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parents[1] / "0_Shared"))
    import nc_io

----------------------------------------------------------------------------
PROJECT NC CONVENTION (the "standard" this module enforces)
----------------------------------------------------------------------------
1. COORDINATE NAMES
   Always ``lat`` / ``lon`` (and ``time`` if temporal). Variants such as
   latitude/longitude/x/y/valid_time are renamed on the way in.

2. ORIENTATION  (taken from ERA5-Land as delivered by the CDS)
   lat : DESCENDING  (north → south)
   lon : ASCENDING   (west  → east)
   Every written file is forced to this order, so all datasets share the
   same orientation. The order of an existing file can be checked with
   nc_io.detect_orientation(path).

3. FILL VALUE  (the "NaN value")
   Every data variable gets an explicit ``_FillValue``:
     - float variables → NaN
     - integer / flag variables → an explicit integer (e.g. 0 = NoData)
   Coordinate variables (lat/lon/time) get ``_FillValue = None`` so they are
   never assigned a fill value (CF requirement).

4. COMPRESSION
   zlib, complevel 4, for every data variable (overridable).

5. GLOBAL METADATA — required keys on every final file:
   title, institution, source, references, Conventions ("CF-1.8"),
   history, created, crs.
   Plus geospatial bounds, auto-filled from the data.

6. FOLDER TIERS
   Every file this module touches must live under one of:
     1_Original  →  2_Intermediate  →  3_Final
   Final products additionally follow  3_Final/<DatasetName>/<file>.nc .
----------------------------------------------------------------------------
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import xarray as xr

# =============================================================================
# CONVENTION CONSTANTS  — edit here once, applies everywhere
# =============================================================================
LAT_ORDER = "descending"   # ERA5-Land orientation
LON_ORDER = "ascending"

TIER_FOLDERS = ("1_Original", "2_Intermediate", "3_Final")

DEFAULT_COMPLEVEL = 4

# Global attributes that a *final* file is expected to carry.
REQUIRED_GLOBAL_ATTRS = (
    "title", "institution", "source", "references",
    "Conventions", "history", "created", "crs",
)

# Standard coordinate attributes (CF).
_COORD_ATTRS = {
    "lat":  {"long_name": "latitude",  "standard_name": "latitude",
             "units": "degrees_north", "axis": "Y"},
    "lon":  {"long_name": "longitude", "standard_name": "longitude",
             "units": "degrees_east",  "axis": "X"},
    "time": {"long_name": "time",      "standard_name": "time", "axis": "T"},
}

_RENAME_MAP = [("latitude", "lat"), ("longitude", "lon"),
               ("y", "lat"), ("x", "lon"), ("valid_time", "time")]


# =============================================================================
# COORDINATES
# =============================================================================
def normalize_coords(ds: xr.Dataset) -> xr.Dataset:
    """Rename latitude/longitude/x/y/valid_time → lat/lon/time."""
    rename = {old: new for old, new in _RENAME_MAP
              if old in ds.dims or old in ds.coords}
    return ds.rename(rename) if rename else ds


def detect_orientation(obj) -> dict[str, str]:
    """
    Report the lat/lon order of a file or Dataset.
    Accepts a path (str/Path) or an open xarray.Dataset.

        >>> nc_io.detect_orientation("…/era5_land.nc")
        {'lat': 'descending', 'lon': 'ascending'}
    """
    close = False
    if isinstance(obj, (str, Path)):
        obj = normalize_coords(xr.open_dataset(obj, decode_times=False))
        close = True
    out = {}
    for c in ("lat", "lon"):
        if c in obj.coords and obj.sizes.get(c, 1) > 1:
            v = obj[c].values
            out[c] = "ascending" if v[0] < v[-1] else "descending"
        else:
            out[c] = "n/a"
    if close:
        obj.close()
    return out


def enforce_orientation(ds: xr.Dataset,
                        lat_order: str = LAT_ORDER,
                        lon_order: str = LON_ORDER) -> xr.Dataset:
    """Sort lat/lon to match the project convention (no-op if already ordered)."""
    if "lat" in ds.coords and ds.sizes.get("lat", 1) > 1:
        is_asc   = bool(ds.lat.values[0] < ds.lat.values[-1])
        want_asc = (lat_order == "ascending")
        if is_asc != want_asc:
            ds = ds.sortby("lat", ascending=want_asc)
    if "lon" in ds.coords and ds.sizes.get("lon", 1) > 1:
        is_asc   = bool(ds.lon.values[0] < ds.lon.values[-1])
        want_asc = (lon_order == "ascending")
        if is_asc != want_asc:
            ds = ds.sortby("lon", ascending=want_asc)
    return ds


def assign_coord_attrs(ds: xr.Dataset) -> xr.Dataset:
    """Stamp standard CF attributes onto lat/lon/time (keeps existing extras)."""
    for c, attrs in _COORD_ATTRS.items():
        if c in ds.coords:
            merged = {**attrs, **ds[c].attrs}   # don't clobber dataset-specific extras
            ds[c].attrs = merged
    return ds


# =============================================================================
# FOLDER-TIER VALIDATION
# =============================================================================
def which_tier(path: str | Path) -> str | None:
    """Return the tier folder ('1_Original'…) a path lives under, else None."""
    parts = set(Path(path).parts)
    for tier in TIER_FOLDERS:
        if tier in parts:
            return tier
    return None


def check_tier(path: str | Path, *, require_final_subfolder: bool = True) -> str:
    """
    Verify a path sits inside a known tier. Raise on violation.
    For 3_Final, also warn if the file is dumped directly in 3_Final/ instead
    of 3_Final/<DatasetName>/<file>.nc.
    """
    path = Path(path)
    tier = which_tier(path)
    if tier is None:
        raise ValueError(
            f"Path is not inside a tier folder {TIER_FOLDERS}:\n  {path}"
        )
    if tier == "3_Final" and require_final_subfolder:
        idx = path.parts.index("3_Final")
        # expect …/3_Final/<DatasetName>/<file>.nc  → at least 2 parts after 3_Final
        if len(path.parts) - idx < 3:
            print(f"  ⚠  {path.name} is directly in 3_Final/ — convention is "
                  f"3_Final/<DatasetName>/{path.name}")
    return tier


# =============================================================================
# METADATA
# =============================================================================
def make_global_attrs(*, title: str, source: str,
                       institution: str = "not specified",
                       references: str = "",
                       crs: str = "EPSG:4326 (WGS84)",
                       script: str | None = None,
                       extra: Mapping | None = None) -> dict:
    """Build a complete, convention-compliant global-attribute block."""
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    attrs = {
        "title":       title,
        "institution": institution,
        "source":      source,
        "references":  references,
        "Conventions": "CF-1.8",
        "history":     f"{ts}: written by {script or 'nc_io.save_nc'}",
        "created":     ts,
        "crs":         crs,
    }
    if extra:
        attrs.update(extra)
    return attrs


def add_geospatial_bounds(ds: xr.Dataset, attrs: dict) -> dict:
    """Fill geospatial_* and time_coverage_* from the data."""
    if "lat" in ds.coords:
        attrs["geospatial_lat_min"] = float(ds.lat.min())
        attrs["geospatial_lat_max"] = float(ds.lat.max())
    if "lon" in ds.coords:
        attrs["geospatial_lon_min"] = float(ds.lon.min())
        attrs["geospatial_lon_max"] = float(ds.lon.max())
    if "time" in ds.coords and ds.sizes.get("time", 0) > 0:
        attrs["time_coverage_start"] = str(np.asarray(ds.time.values)[0])
        attrs["time_coverage_end"]   = str(np.asarray(ds.time.values)[-1])
    return attrs


def validate_metadata(ds: xr.Dataset, *, required: Sequence[str] = REQUIRED_GLOBAL_ATTRS):
    """Warn about missing required global attrs and per-variable units/long_name."""
    missing = [k for k in required if k not in ds.attrs]
    if missing:
        print(f"  ⚠  Missing global attrs: {missing}")
    for v in ds.data_vars:
        a = ds[v].attrs
        if "long_name" not in a:
            print(f"  ⚠  Variable '{v}' has no long_name")
        is_flag = "flag_values" in a or "flag_meanings" in a
        if "units" not in a and not is_flag:
            print(f"  ⚠  Variable '{v}' has no units")


# =============================================================================
# ENCODING (fill value, dtype, compression)
# =============================================================================
def build_encoding(ds: xr.Dataset, *,
                   fill_value=np.nan,
                   dtype=None,
                   compress: bool = True,
                   complevel: int = DEFAULT_COMPLEVEL,
                   suppress_coord_fill: bool = True) -> dict:
    """
    Build a complete encoding dict.

    fill_value : scalar applied to all vars, OR dict {var: value}.
                 Floats default to NaN; integer vars MUST get an integer fill.
    dtype      : None (preserve) | numpy dtype for all | dict {var: dtype}.
    """
    enc: dict = {}
    for var in ds.data_vars:
        fv = fill_value.get(var, np.nan) if isinstance(fill_value, dict) else fill_value
        dt = dtype.get(var) if isinstance(dtype, dict) else dtype

        target_kind = np.dtype(dt).kind if dt is not None else ds[var].dtype.kind
        e: dict = {}

        if fv is not None:
            is_nan = isinstance(fv, float) and np.isnan(fv)
            if target_kind in "iu" and is_nan:
                raise ValueError(
                    f"'{var}' is an integer variable but fill_value is NaN. "
                    f"Pass an explicit integer, e.g. fill_value={{'{var}': 0}}."
                )
            e["_FillValue"] = fv
        if dt is not None:
            e["dtype"] = np.dtype(dt)
        if compress:
            e["zlib"] = True
            e["complevel"] = complevel
        enc[var] = e

    if suppress_coord_fill:
        for c in ("lat", "lon", "time"):
            if c in ds.coords:
                enc.setdefault(c, {})["_FillValue"] = None
    return enc


# =============================================================================
# THE ONE-CALL SAVER
# =============================================================================
def save_nc(ds: xr.Dataset, path: str | Path, *,
            fill_value=np.nan,
            dtype=None,
            global_attrs: Mapping | None = None,
            compress: bool = True,
            complevel: int = DEFAULT_COMPLEVEL,
            enforce_orient: bool = True,
            require_tier: bool = True,
            stamp_coord_attrs: bool = True,
            encoding_overrides: Mapping | None = None,
            verify: bool = True) -> Path:
    """
    Write a Dataset the project-standard way:
      tier check → coord normalize → orientation → coord attrs →
      geospatial bounds → encoding (fill/dtype/zlib, coord fill suppressed) →
      write → metadata validation → optional reopen verification.

    Returns the output Path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if require_tier:
        check_tier(path)

    ds = normalize_coords(ds)
    if enforce_orient:
        ds = enforce_orientation(ds)
    if stamp_coord_attrs:
        ds = assign_coord_attrs(ds)

    if global_attrs is not None:
        ds.attrs = add_geospatial_bounds(ds, dict(global_attrs))

    validate_metadata(ds)

    # Clear any stale per-variable encoding (e.g. source scale_factor/add_offset
    # or int16 packing from the CDS, or rioxarray nodata) so our explicit
    # encoding fully controls the output. Also drop _FillValue/missing_value
    # from attrs, since xarray forbids them living in both attrs and encoding.
    for _v in list(ds.variables):
        ds[_v].encoding = {}
        ds[_v].attrs.pop("_FillValue", None)
        ds[_v].attrs.pop("missing_value", None)

    enc = build_encoding(ds, fill_value=fill_value, dtype=dtype,
                         compress=compress, complevel=complevel)

    # Per-script escape hatch, e.g. explicit CF time encoding:
    #   encoding_overrides={"time": {"dtype": "float64",
    #                                "units": "days since 1970-01-01",
    #                                "calendar": "proleptic_gregorian"}}
    if encoding_overrides:
        for _k, _v in encoding_overrides.items():
            enc.setdefault(_k, {}).update(_v)

    print(f"  Saving → {path}")
    ds.to_netcdf(path, encoding=enc)
    print(f"  File size : {path.stat().st_size / 1e6:.2f} MB")

    if verify:
        with xr.open_dataset(path, decode_cf=False) as chk:
            print("  Verify:")
            print(f"    orientation : {detect_orientation(chk)}")
            for v in chk.data_vars:
                fv = chk[v].attrs.get("_FillValue", "—")
                print(f"    {v:<22} dtype={str(chk[v].dtype):<8} _FillValue={fv}")
    return path


# =============================================================================
# READER-SIDE HELPERS  (fill-value introspection for feature extraction)
# =============================================================================
# The save side (above) writes an explicit _FillValue on every variable:
# float vars -> NaN, integer/flag vars -> an explicit integer sentinel.
# The extraction scripts must read that declared fill back so they can build a
# correct validity mask, instead of assuming "missing == NaN" (which silently
# breaks for integer/categorical vars whose sentinel is e.g. 0 or 255).
#
# These helpers are the single source of truth the extractor uses to decide
# which cells are valid — the same mask then drives both the nearest-valid
# radius search and the per-(well x raster) provenance flags.

def _fill_is_nan(fv) -> bool:
    """True only for a genuine floating NaN fill value."""
    if fv is None:
        return False
    try:
        return bool(np.isnan(fv))
    except (TypeError, ValueError):
        return False


def read_fill_values(obj, *, include_coords: bool = False) -> dict:
    """
    Return ``{var: declared _FillValue}`` read straight from the file/Dataset,
    WITHOUT relying on xarray's mask-and-scale decoding.

    Accepts a path (str/Path) or an open xarray.Dataset. For a path the file is
    opened with ``decode_cf=False, mask_and_scale=False`` so the raw declared
    fill (and any integer sentinel) is visible in the variable attributes.

    Looks in ``attrs['_FillValue']`` first, then ``attrs['missing_value']``,
    then the same keys in ``.encoding`` (covers Datasets opened decoded).
    A value of ``None`` means the variable declares no fill on disk.
    """
    close = False
    if isinstance(obj, (str, Path)):
        obj = xr.open_dataset(obj, decode_cf=False, mask_and_scale=False)
        close = True

    names = list(obj.data_vars)
    if include_coords:
        names += [c for c in obj.coords if c not in names]

    out: dict = {}
    for v in names:
        a = obj[v].attrs
        e = getattr(obj[v], "encoding", {}) or {}
        fv = a.get("_FillValue", a.get("missing_value",
                   e.get("_FillValue", e.get("missing_value", None))))
        out[v] = fv

    if close:
        obj.close()
    return out


def valid_mask(values, fill_value) -> np.ndarray:
    """
    Boolean array, True where a cell holds REAL data (not the fill, not NaN).

    Handles all three convention cases uniformly:
      - fill_value is None           -> only NaN is invalid (floats); ints all valid
      - fill_value is NaN            -> invalid == not finite
      - fill_value is an int sentinel-> invalid == equal to sentinel (+ NaN for floats)

    This is the mask the radius search treats as "valid pixels".
    """
    values = np.asarray(values)
    is_float = np.issubdtype(values.dtype, np.floating)

    if fill_value is None or _fill_is_nan(fill_value):
        return np.isfinite(values) if is_float else np.ones(values.shape, dtype=bool)

    mask = values != fill_value
    if is_float:
        mask &= np.isfinite(values)
    return mask


def report_fills(obj, *, label: str | None = None) -> dict:
    """
    Print a per-variable table of dtype + declared _FillValue and return the
    ``{var: fill}`` dict. Warns when an integer/flag variable declares no fill
    (a likely convention violation that would let a sentinel leak in as data).
    Reads metadata only — does not load any data arrays.
    """
    close = False
    src = obj
    if isinstance(obj, (str, Path)):
        src = xr.open_dataset(obj, decode_cf=False, mask_and_scale=False)
        close = True

    fills = read_fill_values(src)
    if label:
        print(f"  Fill values — {label}")
    print(f"    {'variable':<26} {'dtype':<10} {'_FillValue'}")
    n_warn = 0
    for v, fv in fills.items():
        dt   = str(src[v].dtype)
        kind = src[v].dtype.kind
        if _fill_is_nan(fv):
            shown = "NaN"
        elif fv is None:
            shown = "—"
        else:
            shown = fv
        warn = ""
        if kind in "iu" and fv is None:
            warn = "   <-- integer var without declared _FillValue"
            n_warn += 1
        print(f"    {v:<26} {dt:<10} {shown}{warn}")
    if n_warn:
        print(f"    ({n_warn} integer variable(s) without a declared fill "
              f"— extractor will treat every value as valid for those)")

    if close:
        src.close()
    return fills
