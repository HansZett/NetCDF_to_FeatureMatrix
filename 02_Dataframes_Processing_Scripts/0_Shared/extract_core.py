# -*- coding: utf-8 -*-
"""
extract_core.py — manifest-driven point extraction engine
=========================================================

ONE implementation of "sample a grid at well locations" that both the static
and the dynamic store builders call. The behaviour per dataset (search radius,
what to do without a valid cell) is parameterised by
`dataset_manifest.DatasetSpec`.

Location: 02_Dataframes_Processing_Scripts/0_Shared/extract_core.py
          (imported by the scripts in 2_Feature_Stores/)

What it does per (well x dataset)
---------------------------------
1. Strict nearest grid cell (vectorised, handles descending lat).
2. If radius>0: search the neighbourhood for the CLOSEST VALID cell, where
   "valid" comes from nc_io.valid_mask using the variable's DECLARED _FillValue
   (so an integer sentinel like 255 / 0 is never sampled as real data).
3. One cell is chosen per (well, dataset) from a reference variable, then every
   variable is read at that same cell -> coherent provenance.
4. When no valid cell is within the radius, `on_no_valid` decides:
   keep_nan -> NaN (well kept) | skip_well -> well dropped | nearest_global ->
   nearest valid with no distance limit.

Provenance per (well x dataset)  [one row, NOT per month]
--------------------------------
   sample_dist_km : great-circle distance well -> chosen cell centre
   rescued        : chosen cell != strict nearest cell
   outcome        : 'home' | 'rescued' | 'nearest_global' | 'no_valid' | 'dropped'
   keep           : False only for skip_well + no valid in radius

Values are returned as float64 with NaN for invalid/fill (categorical class
codes included, e.g. 30.0) so a single dtype carries missingness uniformly;
casting back to Int/category happens at model time.

CLI self-check (runs on the real well locations, writes nothing):
    python extract_core.py --key era5l
    python extract_core.py --key twsa  --sample 8
    python extract_core.py --key soilgrids
    python extract_core.py --key soilgrids --root "<path to project root>"
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import xarray as xr

# =============================================================================
# PATHS  (relative to the project root; adjust here if the layout changes)
# =============================================================================
_HERE = Path(__file__).resolve().parent
_ROOT = Path(__file__).resolve().parents[2]
# nc_io lives with the NetCDF pipeline (02_Processing_Scripts/0_Shared); fall
# back to this folder if a copy of nc_io.py sits next to these modules.
NC_IO_DIR = _ROOT / "02_Processing_Scripts" / "0_Shared"
# Well metadata (MW_ID, lon, lat) used by the CLI self-check below
DEFAULT_META = ("03_Dataframes", "02_Static_Features", "01_GEMS-GER",
                "02_adapted_well_metadata.csv")

for _cand in (NC_IO_DIR, _HERE):
    if (_cand / "nc_io.py").exists():
        sys.path.append(str(_cand))
        break
sys.path.append(str(_HERE))
import nc_io                         # noqa: E402
import dataset_manifest as dm        # noqa: E402

EARTH_R_KM = 6371.0088


# =============================================================================
# GEOMETRY
# =============================================================================
def haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Great-circle distance in km. Inputs broadcast; degrees in."""
    lat1, lon1, lat2, lon2 = map(np.asarray, (lat1, lon1, lat2, lon2))
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlmb = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return 2 * EARTH_R_KM * np.arcsin(np.sqrt(a))


def nearest_1d(grid: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """
    Index of the closest grid coordinate for each target.
    Works for ascending or descending, regular or irregular 1-D coords.
    """
    grid = np.asarray(grid, dtype="float64")
    targets = np.asarray(targets, dtype="float64")
    if grid.size <= 1:                       # single-cell axis -> only index 0
        return np.zeros(targets.shape, dtype="int64")
    asc = grid[0] < grid[-1]
    g = grid if asc else grid[::-1]
    pos = np.searchsorted(g, targets)
    pos = np.clip(pos, 1, len(g) - 1)
    left = g[pos - 1]
    right = g[pos]
    choose_left = (targets - left) <= (right - targets)
    idx = np.where(choose_left, pos - 1, pos)
    if not asc:
        idx = (len(grid) - 1) - idx
    return idx.astype("int64")


# =============================================================================
# PIXEL SELECTION  (the heart; returns provenance)
# =============================================================================
def choose_pixels(lat_grid, lon_grid, well_lats, well_lons, valid2d, *,
                  radius_km: float, on_no_valid: str,
                  max_half_window: int = 60) -> dict:
    """
    Choose one grid cell per well from a 2-D validity mask (lat, lon).

    Returns arrays of length n_wells: idx_y, idx_x (int, -1 if dropped),
    dist_km, rescued (bool), outcome (object), keep (bool).
    """
    lat_grid = np.asarray(lat_grid, dtype="float64")
    lon_grid = np.asarray(lon_grid, dtype="float64")
    well_lats = np.asarray(well_lats, dtype="float64")
    well_lons = np.asarray(well_lons, dtype="float64")
    n = len(well_lats)
    ny, nx = valid2d.shape

    iy0 = nearest_1d(lat_grid, well_lats)        # strict nearest
    ix0 = nearest_1d(lon_grid, well_lons)

    iy = iy0.copy()
    ix = ix0.copy()
    dist = haversine_km(well_lats, well_lons, lat_grid[iy0], lon_grid[ix0])
    outcome = np.array(["home"] * n, dtype=object)
    keep = np.ones(n, dtype=bool)

    home_valid = valid2d[iy0, ix0]

    # cells per degree -> window size needed to cover the radius
    dlat = float(np.abs(np.median(np.diff(lat_grid)))) if ny > 1 else 1.0
    dlon = float(np.abs(np.median(np.diff(lon_grid)))) if nx > 1 else 1.0
    km_per_deg_lat = 111.32
    half_y = 0 if radius_km <= 0 else int(np.ceil(radius_km / (dlat * km_per_deg_lat)))
    # lon shrinks with latitude; use the AOI's max |lat| for a safe window
    cos_lat = max(np.cos(np.radians(np.abs(well_lats).max())), 0.1)
    half_x = 0 if radius_km <= 0 else int(np.ceil(radius_km / (dlon * km_per_deg_lat * cos_lat)))
    half_y = min(half_y, max_half_window)
    half_x = min(half_x, max_half_window)

    need = np.where(~home_valid)[0]   # only these wells need a search

    for w in need:
        cy, cx = iy0[w], ix0[w]
        best_d, best_y, best_x = np.inf, -1, -1
        if radius_km > 0:
            y0, y1 = max(0, cy - half_y), min(ny, cy + half_y + 1)
            x0, x1 = max(0, cx - half_x), min(nx, cx + half_x + 1)
            sub = valid2d[y0:y1, x0:x1]
            if sub.any():
                yy, xx = np.nonzero(sub)
                gy, gx = yy + y0, xx + x0
                d = haversine_km(well_lats[w], well_lons[w],
                                 lat_grid[gy], lon_grid[gx])
                within = d <= radius_km
                if within.any():
                    k = np.argmin(np.where(within, d, np.inf))
                    best_d, best_y, best_x = d[k], gy[k], gx[k]

        if best_y >= 0:                                  # rescued a neighbour
            iy[w], ix[w], dist[w] = best_y, best_x, best_d
            outcome[w] = "rescued"
        else:                                            # nothing valid in radius
            if on_no_valid == "nearest_global":
                yy, xx = np.nonzero(valid2d)
                if len(yy):
                    d = haversine_km(well_lats[w], well_lons[w],
                                     lat_grid[yy], lon_grid[xx])
                    k = int(np.argmin(d))
                    iy[w], ix[w], dist[w] = yy[k], xx[k], d[k]
                    outcome[w] = "nearest_global"
                else:
                    iy[w], ix[w], dist[w], keep[w], outcome[w] = -1, -1, np.nan, False, "no_valid"
            elif on_no_valid == "skip_well":
                iy[w], ix[w], dist[w], keep[w], outcome[w] = -1, -1, np.nan, False, "dropped"
            else:  # keep_nan
                iy[w], ix[w], dist[w], outcome[w] = -1, -1, np.nan, "no_valid"

    rescued = outcome == "rescued"
    return {"idx_y": iy, "idx_x": ix, "dist_km": dist,
            "rescued": rescued, "outcome": outcome, "keep": keep}


# =============================================================================
# VALUE READING
# =============================================================================
def _spatial_dims(da, lat_name="lat", lon_name="lon"):
    return [d for d in da.dims if d not in (lat_name, lon_name)]


def _read_var_at_cells(da: xr.DataArray, iy, ix, fill) -> tuple:
    """
    Read `da` at the chosen (iy, ix) cells. Returns (column_suffixes, values):
      - static var (lat,lon)            -> [""],            array (n_wells,)
      - static var with extra dim (d,..)-> ["_<dimval>",..],array (n_extra, n_wells)
      - dynamic var (time,lat,lon)      -> [""],            array (n_time, n_wells)
    Invalid/fill cells -> NaN. Output is float32.  iy/ix == -1 (dropped) -> NaN.

    Sampling happens on the RAW array BEFORE any float cast, so the full grid is
    never promoted to float (critical for LCC: 31 yr x 300 m would be GBs). Only
    the tiny gathered (n_wells) slice is cast.
    """
    extra = [d for d in _spatial_dims(da) if d != "time"]
    da = da.transpose(*(["time"] if "time" in da.dims else []),
                      *extra, "lat", "lon")
    raw = np.asarray(da.values)                  # raw dtype, NOT cast

    n = len(iy)
    safe_y = np.where(iy >= 0, iy, 0)
    safe_x = np.where(ix >= 0, ix, 0)
    dropped = iy < 0

    g = raw[..., safe_y, safe_x]                 # (..., n_wells)  raw dtype
    valid = nc_io.valid_mask(g, fill)            # mask the small slice only
    g = g.astype("float32")
    g[~valid] = np.nan

    if extra:
        suff_vals = [da[d].values for d in extra]
        if len(extra) == 1:
            suffixes = [f"_{_fmt(v)}" for v in suff_vals[0]]
        else:
            import itertools
            suffixes = ["_" + "_".join(_fmt(v) for v in combo)
                        for combo in itertools.product(*suff_vals)]
    else:
        suffixes = [""]

    has_time = "time" in da.dims
    if has_time:
        g = g.reshape(raw.shape[0], -1, n)       # (time, n_extra, n_wells)
        g[:, :, dropped] = np.nan
        g = np.moveaxis(g, 1, 0)                 # (n_extra, time, n_wells)
        out = [g[e] for e in range(g.shape[0])]
    else:
        g = g.reshape(-1, n)                     # (n_extra, n_wells)
        g[:, dropped] = np.nan
        out = [g[e] for e in range(g.shape[0])]
    return suffixes, out


def _fmt(v) -> str:
    """Compact label for an extra-dim coordinate value (e.g. depth)."""
    try:
        f = float(v)
        return f"{f:g}".replace(".", "p").replace("-", "m")
    except (TypeError, ValueError):
        return str(v).strip().replace(" ", "")


# =============================================================================
# TOP-LEVEL: extract one dataset for all wells
# =============================================================================
def extract_dataset(spec: dm.DatasetSpec, well_ids, well_lons, well_lats, *,
                    root=None, verbose=True) -> dict:
    """
    Returns:
      {
        'key': str, 'temporal': str,
        'times': np.ndarray | None,                 # dynamic only
        'columns': {col_name: values},              # static: (n_wells,)
                                                    # dynamic: (n_time, n_wells)
        'prov': {idx_y, idx_x, dist_km, rescued, outcome, keep},
      }
    Column names are '<key>__<var><suffix>'.
    """
    path = dm.nc_path(spec, root)
    if path is None:
        raise ValueError(f"{spec.key}: basin source has no single NC; "
                         f"use the basin path instead.")

    # mask_and_scale=False keeps raw fills/integers; decode_times stays on.
    ds = xr.open_dataset(path, mask_and_scale=False)
    ds = nc_io.normalize_coords(ds)
    for dv in spec.drop_vars:
        if dv in ds.variables:
            ds = ds.drop_vars(dv)

    var_list = spec.vars or [v for v in ds.data_vars]
    var_list = [v for v in var_list if v in ds.data_vars]
    if not var_list:
        raise ValueError(f"{spec.key}: none of the requested vars present in {path.name}")

    # ---- bbox subset around the wells (+ margin >= radius). Identical results,
    #      just limits how much grid is read — essential for fine grids (LCC).
    #      NOTE: with this subset, on_no_valid='nearest_global' is bounded by the
    #      bbox rather than the whole globe; no default dataset uses it.
    radius = dm.resolved_radius_km(spec)
    res_deg = spec.resolution_deg or 0.1
    margin = radius / dm.KM_PER_DEG + 3.0 * res_deg   # >= a few cells; never collapses
    well_lats = np.asarray(well_lats, float)
    well_lons = np.asarray(well_lons, float)
    lat_desc = bool(ds["lat"].values[0] > ds["lat"].values[-1])
    lat_slice = (slice(float(well_lats.max()) + margin, float(well_lats.min()) - margin)
                 if lat_desc else
                 slice(float(well_lats.min()) - margin, float(well_lats.max()) + margin))
    ds = ds.sel(lat=lat_slice,
                lon=slice(float(well_lons.min()) - margin,
                          float(well_lons.max()) + margin))

    fills = nc_io.read_fill_values(ds)
    lat_grid = ds["lat"].values
    lon_grid = ds["lon"].values

    # ---- reference validity mask (2-D): first var, first time slice if dynamic
    ref = var_list[0]
    ref_da = ds[ref]
    has_time = "time" in ref_da.dims
    ref2d = ref_da.isel(time=0) if has_time else ref_da
    # collapse any remaining extra dim for the validity reference
    for d in _spatial_dims(ref2d):
        if d != "time":
            ref2d = ref2d.isel({d: 0})
    valid2d = nc_io.valid_mask(np.asarray(ref2d.values), fills.get(ref))

    prov = choose_pixels(lat_grid, lon_grid, well_lats, well_lons, valid2d,
                         radius_km=radius, on_no_valid=spec.on_no_valid)

    if verbose:
        oc = prov["outcome"]
        n = len(well_ids)
        counts = {k: int((oc == k).sum()) for k in
                  ("home", "rescued", "nearest_global", "no_valid", "dropped")}
        d_ok = prov["dist_km"][np.isfinite(prov["dist_km"])]
        print(f"  [{spec.key}] grid {valid2d.shape}, radius {radius:g} km, "
              f"ref-var '{ref}', on_no_valid={spec.on_no_valid}")
        print("    outcomes: " + "  ".join(f"{k}={v}" for k, v in counts.items()
                                            if v) + f"   (of {n} wells)")
        if len(d_ok):
            print(f"    sample_dist_km: median={np.median(d_ok):.2f}  "
                  f"p95={np.percentile(d_ok, 95):.2f}  max={d_ok.max():.2f}")

    # ---- read every variable at the chosen cells
    times = ds["time"].values if has_time else None
    columns: dict = {}
    for v in var_list:
        suffixes, arrs = _read_var_at_cells(ds[v], prov["idx_y"], prov["idx_x"],
                                            fills.get(v))
        for suf, a in zip(suffixes, arrs):
            columns[f"{spec.key}__{v}{suf}"] = a

    ds.close()
    return {"key": spec.key, "temporal": spec.temporal, "times": times,
            "columns": columns, "prov": prov, "well_ids": list(well_ids)}


# =============================================================================
# CLI SELF-CHECK
# =============================================================================
def _load_wells(root: Path, meta_rel):
    import pandas as pd
    p = root.joinpath(*meta_rel)
    if not p.exists():
        raise FileNotFoundError(p)
    df = pd.read_csv(p)
    id_col = "MW_ID" if "MW_ID" in df.columns else df.columns[0]
    df = df.dropna(subset=["lon", "lat"]).reset_index(drop=True)
    ids = df[id_col].astype(str).str.replace("MW_", "", regex=False).str.strip()
    return ids.tolist(), df["lon"].to_numpy(float), df["lat"].to_numpy(float)


if __name__ == "__main__":
    import argparse
    import pandas as pd

    ap = argparse.ArgumentParser()
    ap.add_argument("--key", required=True, help="dataset key from the manifest")
    ap.add_argument("--root", default=None, help="project root (overrides default)")
    ap.add_argument("--sample", type=int, default=6, help="wells to preview")
    args = ap.parse_args()

    root = Path(args.root).resolve() if args.root else dm.PROJECT_ROOT
    spec = dm.BY_KEY.get(args.key)
    if spec is None:
        sys.exit(f"unknown key '{args.key}'. options: {', '.join(dm.BY_KEY)}")
    if spec.mode == "basin":
        sys.exit(f"'{args.key}' is a basin dataset — handled by the basin path, "
                 f"not this point engine.")

    print(f"  project root : {root}")
    ids, lons, lats = _load_wells(root, DEFAULT_META)
    print(f"  wells        : {len(ids)}")

    res = extract_dataset(spec, ids, lons, lats, root=root, verbose=True)

    prov = res["prov"]
    cols = res["columns"]
    print(f"\n  produced {len(cols)} feature column(s); "
          f"{'time x wells' if res['temporal']!='static' else 'wells'}")
    if res["times"] is not None:
        print(f"  time axis    : {str(res['times'][0])[:10]} .. "
              f"{str(res['times'][-1])[:10]}  ({len(res['times'])} steps)")

    # preview
    k = min(args.sample, len(ids))
    show_cols = list(cols)[:3]
    print(f"\n  sample of {k} wells (cols shown: {show_cols}):")
    hdr = f"    {'well':<10}{'outcome':<10}{'dist_km':>9}  rescued  " + \
          "  ".join(f"{c.split('__')[-1]:>14}" for c in show_cols)
    print(hdr)
    for i in range(k):
        vals = []
        for c in show_cols:
            a = cols[c]
            x = a[0, i] if res["temporal"] != "static" else a[i]
            vals.append(f"{x:>14.4g}" if np.isfinite(x) else f"{'NaN':>14}")
        print(f"    {ids[i]:<10}{prov['outcome'][i]:<10}"
              f"{prov['dist_km'][i]:>9.2f}  {str(bool(prov['rescued'][i])):<7}  "
              + "  ".join(vals))

    # provenance summary table (one row per well x dataset is what we'd store)
    prov_df = pd.DataFrame({
        "well_id": [f"MW_{w}" for w in ids],
        f"{spec.key}__sample_dist_km": np.round(prov["dist_km"], 3),
        f"{spec.key}__rescued": prov["rescued"],
        f"{spec.key}__outcome": prov["outcome"],
    })
    print("\n  provenance (one row per well) — first rows:")
    print(prov_df.head(k).to_string(index=False))
    print("\n  Done ✅  (nothing written — diagnostic only)")
