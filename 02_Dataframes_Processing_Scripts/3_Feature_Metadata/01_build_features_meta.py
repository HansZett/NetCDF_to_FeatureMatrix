# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""
01_build_features_meta.py

Builds the feature metadata tables for both store profiles:
  - features_meta_global.csv   : own features extracted from the global
                                 NetCDF datasets (static and dynamic store
                                 built in 2_Feature_Stores/).
  - features_meta_GEMS-GER.csv : published GEMS-GER dataset (tables only, no
                                 NetCDF files -> hand-curated in GEMS_METADATA).

The two tables are NOT merged: they are maintained differently (global =
enumerated automatically from the store headers; GEMS-GER = hand-curated)
and some column names exist in both stores (e.g. 'Elevation'). Every
experiment reads exactly ONE of the tables (store_profile).

Columns of the metadata tables:
  feature_raw_name : column name in the respective store
  source_system    : 'global' | 'gems'
  type             : 'dynamic' | 'static'
  group            : machine key (prefix before '__'; C3S__ columns -> 'lcc')
  var_group        : coarse group for SHAP aggregation / plot colour (from the
                     overrides; falls back to 'group')
  processing       : 'lag_roll' | 'passthrough'
  categorical      : bool (independent of processing)
  temporal         : 'monthly' | 'static'
  time_start       : 'YYYY-MM' or '' (static)
  time_end         : 'YYYY-MM' or '' (static); shows the shorter TWSA period
  plot_name        : readable label for plots (from the overrides, otherwise
                     derived from the column name)
  unit             : unit (from the overrides; '?' if not set)

Input : 03_Dataframes/01_Dynamic_Features/03_global_monthly/MW_<id>.csv (header)
        03_Dataframes/02_Static_Features/02_global/static_features.csv   (header)
        03_Dataframes/01_Dynamic_Features/02_GEMS-GER_monthly/MW_<id>_monthly.csv
        03_Dataframes/02_Static_Features/01_GEMS-GER/01_original_well_metadata.csv
        feature_labels_overrides.csv                  (next to this script)
Output: 03_Dataframes/03_Metadata/features_meta_global.csv
        03_Dataframes/03_Metadata/features_meta_GEMS-GER.csv
        (semicolon-separated)

Location: 02_Dataframes_Processing_Scripts/3_Feature_Metadata/
Run     : python 01_build_features_meta.py
          (after 2_Feature_Stores/01 and 02, because the store headers
           define the feature columns)
"""

import os
import pandas as pd

# ==========================================
# 1. PATHS  (relative to the project root; adjust here if the layout changes)
# ==========================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

# Own (global) feature stores
pth_my_dyn_store  = os.path.join(PROJECT_ROOT, "03_Dataframes", "01_Dynamic_Features", "03_global_monthly")
pth_my_stat_store = os.path.join(PROJECT_ROOT, "03_Dataframes", "02_Static_Features", "02_global", "static_features.csv")

# GEMS-GER stores (published benchmark dataset)
pth_gems_dyn_store  = os.path.join(PROJECT_ROOT, "03_Dataframes", "01_Dynamic_Features", "02_GEMS-GER_monthly")
pth_gems_stat_store = os.path.join(PROJECT_ROOT, "03_Dataframes", "02_Static_Features", "01_GEMS-GER" ,"01_original_well_metadata.csv")

# Curated label overrides (readable names, var_group for SHAP plots, units)
pth_overrides = os.path.join(SCRIPT_DIR, "feature_labels_overrides.csv")

output_dir = os.path.join(PROJECT_ROOT, "03_Dataframes", "03_Metadata")
os.makedirs(output_dir, exist_ok=True)

output_meta_global = os.path.join(output_dir, "features_meta_global.csv")
output_meta_gems   = os.path.join(output_dir, "features_meta_GEMS-GER.csv")

# ==========================================
# 2. CONFIGURATION: store layout (global)
# ==========================================
# Which groups are dynamic (lag/rolling features in the ML script), which
# columns are categorical, and which temporal coverage applies per group.

# LCC columns carry the prefix 'C3S__' in the store (engineered, source C3S /
# ESA CCI land cover). The machine group stays 'lcc' (manifest key), so that
# GROUP_TEMPORAL['lcc'] and downstream group selectors work.
LCC_PREFIX = 'C3S'

# Dynamic groups that receive lag / rolling features in the ML script.
# Basin series carry product-specific prefixes (basin_era5l_levX /
# basin_terra_levX); the shared static basin attributes
# (basin_levX__HYBAS_ID/UP_AREA/COAST) are NOT a lag group.
DYNAMIC_LAG_GROUPS = {
    'era5l', 'terra', 'twsa',
    'basin_era5l_lev06', 'basin_era5l_lev07',
    'basin_terra_lev06', 'basin_terra_lev07',
}

# Temporal coverage per group. twsa ends two years earlier; the column
# time_end makes this visible for the experiment configuration.
GROUP_TEMPORAL = {
    'era5l':             ('monthly', '1991-01', '2022-12'),
    'terra':             ('monthly', '1991-01', '2022-12'),
    'twsa':              ('monthly', '1991-01', '2020-12'),
    'lcc':               ('monthly', '1991-01', '2022-12'),   # yearly, broadcast to months
    'basin_era5l_lev06': ('monthly', '1991-01', '2022-12'),
    'basin_era5l_lev07': ('monthly', '1991-01', '2022-12'),
    'basin_terra_lev06': ('monthly', '1991-01', '2022-12'),
    'basin_terra_lev07': ('monthly', '1991-01', '2022-12'),
    'basin_lev06':       ('static',  '', ''),                 # shared basin identity only
    'basin_lev07':       ('static',  '', ''),
    'era5l_static':      ('static',  '', ''),
    'soilgrids':         ('static',  '', ''),
    'gasy':              ('static',  '', ''),
    'glim':              ('static',  '', ''),
    'whymap':            ('static',  '', ''),
    'hydsheds_terrain':  ('static',  '', ''),
    'hydrivers_dist':    ('static',  '', ''),
    'other':             ('static',  '', ''),                 # well metadata, e.g. Elevation
}
# Note: basin columns are spread over THREE group prefixes:
#   basin_levX__         shared, product-neutral identity/geometry
#                        (HYBAS_ID/UP_AREA/COAST) -> static only
#   basin_era5l_levX__   ERA5-Land fluxes (dynamic) + ERA5 grid diagnostics
#                        (coverage_frac/area, static)
#   basin_terra_levX__   TerraClimate variables (dynamic) + TerraClimate grid
#                        diagnostics (static)
# A product prefix therefore holds static AND dynamic columns. They are NOT
# separated by the group but by ftype ('static' vs 'dynamic') in
# make_global_row -> static => passthrough + temporal 'static'.

# Categorical columns in the global store (independent of processing).
CATEGORICAL_GLOBAL = {
    'era5l_static__slt', 'era5l_static__tvl', 'era5l_static__tvh',
    'glim__litho_class', 'whymap__hygeo2', 'hydsheds_terrain__dir',
    'basin_lev06__COAST', 'basin_lev07__COAST',
    'basin_lev06__HYBAS_ID', 'basin_lev07__HYBAS_ID',
    'AquiferMed',
    'C3S__LCC', 'C3S__LCC_previous',
}


def group_of(col):
    """Machine group of a column. Special case: C3S__ LCC columns -> 'lcc'."""
    if col.startswith(LCC_PREFIX + '__'):
        return 'lcc'
    if '__' in col:
        return col.split('__')[0]
    return 'other'


# ==========================================
# 3. NC UNIT HOOK (not connected)
# ==========================================
def read_nc_unit(group, var):
    """Optional hook to read the unit from the 'units' attribute of the
    variable in its NetCDF file. It is NOT connected and returns None, so
    all units come from feature_labels_overrides.csv ('?' if missing).

    Possible implementation:
        import xarray as xr
        ds = xr.open_dataset(NC_PATH_BY_GROUP[group])
        return ds[var].attrs.get('units')
    """
    return None


# ==========================================
# 4. LABEL OVERRIDES
# ==========================================
def load_overrides(path):
    """Read feature_labels_overrides.csv. Returns three matcher lists
    (exact, base, prefix) of (key, display_label, var_group, unit)."""
    if not os.path.exists(path):
        print(f"Note: no overrides file at {path} -- plot_name is derived from the column name.")
        return {'exact': [], 'base': [], 'prefix': []}
    ov = pd.read_csv(path, sep=';', comment='#', dtype=str).fillna('')
    ov.columns = [c.strip() for c in ov.columns]
    has_unit = 'unit' in ov.columns          # overrides files without a unit column are accepted
    buckets = {'exact': [], 'base': [], 'prefix': []}
    for _, r in ov.iterrows():
        mt = r['match_type'].strip()
        if mt not in buckets:
            continue
        unit = r['unit'].strip() if has_unit else ''
        buckets[mt].append((r['key'].strip(),
                            r['display_label'].strip(),
                            r['var_group'].strip(),
                            unit))
    return buckets


def prettify(col):
    """Fallback label: '__' and '_' -> ' ', first letter upper case."""
    base = col.replace('__', ' ').replace('_', ' ').strip()
    return base[:1].upper() + base[1:] if base else col


def apply_overrides(col, buckets):
    """Find the most specific override for a column.
    Priority: exact > base > prefix. 'base' matches the column itself and
    (later, in the SHAP step) its _lag/_rmean children; in the store it
    behaves like 'exact'. Returns (display_label, var_group, unit)."""
    for key, label, vgrp, unit in buckets['exact']:
        if col == key:
            return label, vgrp, unit
    for key, label, vgrp, unit in buckets['base']:
        if col == key or col.startswith(key + '_'):
            return label, vgrp, unit
    # longest prefix wins
    pref_hits = [(key, label, vgrp, unit) for key, label, vgrp, unit in buckets['prefix']
                 if col.startswith(key)]
    if pref_hits:
        key, label, vgrp, unit = max(pref_hits, key=lambda t: len(t[0]))
        return label, vgrp, unit
    return None, None, None


# ==========================================
# 5. HELPERS (read store headers)
# ==========================================
def get_columns_from_dir(path):
    if not os.path.exists(path):
        print(f"Warning: folder not found: {path}")
        return []
    files = [f for f in os.listdir(path) if f.endswith('.csv') and not f.startswith('_')]
    if not files:
        return []
    df = pd.read_csv(os.path.join(path, sorted(files)[0]), nrows=0)
    return list(df.columns)


def get_columns_from_file(path):
    if not os.path.exists(path):
        print(f"Warning: file not found: {path}")
        return []
    df = pd.read_csv(path, nrows=0)
    return list(df.columns)


# ==========================================
# 6. A. GLOBAL FEATURES (own stores)
# ==========================================
overrides = load_overrides(pth_overrides)
features_global = []

# Columns of the stores that are NOT features (time, target, identifiers,
# coordinates). They get no row in the metadata table.
DROP_GLOBAL_META = {'time', 'GWL', 'MW_ID', 'Proj_ID', 'Operator',
                    'Easting_3035', 'Northing_3035', 'lon', 'lat'}


def make_global_row(col, ftype):
    grp = group_of(col)
    var = col.split('__', 1)[1] if '__' in col else col
    # temporal/processing are set by the SOURCE store (ftype), not only by the
    # prefix: static and dynamic basin columns share a product prefix, so only
    # ftype separates the monthly series (lag_roll) from the static basin
    # attributes (passthrough/static).
    if ftype == 'static':
        temporal, t0, t1 = 'static', '', ''
        proc = 'passthrough'
    else:
        temporal, t0, t1 = GROUP_TEMPORAL.get(grp, ('monthly', '1991-01', '2022-12'))
        proc = 'lag_roll' if grp in DYNAMIC_LAG_GROUPS else 'passthrough'
    is_cat = col in CATEGORICAL_GLOBAL
    label, vgrp, ov_unit = apply_overrides(col, overrides)
    if label is None or label == '':
        label = prettify(col)
    # unit priority: override > NC attribute > '?'
    unit = ov_unit or read_nc_unit(grp, var) or '?'
    return {
        'feature_raw_name': col,
        'source_system': 'global',
        'type': ftype,
        'group': grp,
        'var_group': vgrp if vgrp else grp,
        'processing': proc,
        'categorical': is_cat,
        'temporal': temporal,
        'time_start': t0,
        'time_end': t1,
        'plot_name': label,
        'unit': unit,
    }


print("Reading global dynamic features...")
for col in get_columns_from_dir(pth_my_dyn_store):
    if col in DROP_GLOBAL_META:
        continue
    features_global.append(make_global_row(col, 'dynamic'))

print("Reading global static features...")
for col in get_columns_from_file(pth_my_stat_store):
    if col in DROP_GLOBAL_META:
        continue
    features_global.append(make_global_row(col, 'static'))

df_global = (pd.DataFrame(features_global)
             .drop_duplicates(subset=['feature_raw_name'])
             .reset_index(drop=True))
df_global.to_csv(output_meta_global, index=False, sep=';')
print(f"-> Global metadata table saved ({len(df_global)} features): {output_meta_global}")
n_missing_unit = int((df_global['unit'] == '?').sum())
if n_missing_unit:
    print(f"   Note: {n_missing_unit} features without unit ('?'). "
          f"Add them to feature_labels_overrides.csv or connect read_nc_unit().")


# ==========================================
# 7. B. GEMS-GER FEATURES (hand-curated)
# ==========================================
GEMS_METADATA = {
    'HYRAS_tasmax': {'plot_name': 'Max. Temp (HYRAS)', 'unit': '°C', 'group': 'climate', 'processing': 'lag_roll'},
    'HYRAS_tas':    {'plot_name': 'Mean Temp (HYRAS)', 'unit': '°C', 'group': 'climate', 'processing': 'lag_roll'},
    'HYRAS_tasmin': {'plot_name': 'Min. Temp (HYRAS)', 'unit': '°C', 'group': 'climate', 'processing': 'lag_roll'},
    'HYRAS_pr':     {'plot_name': 'Precipitation (HYRAS)', 'unit': 'mm', 'group': 'climate', 'processing': 'lag_roll'},
    'HYRAS_hurs':   {'plot_name': 'Rel. Humidity (HYRAS)', 'unit': '%', 'group': 'climate', 'processing': 'lag_roll'},
    'DWD_evapo_p':  {'plot_name': 'Pot. Evapotransp. (DWD)', 'unit': 'mm', 'group': 'climate', 'processing': 'lag_roll'},
    'DWD_evapo_r':  {'plot_name': 'Act. Evapotransp. (DWD)', 'unit': 'mm', 'group': 'climate', 'processing': 'lag_roll'},
    'DWD_evapo_fao':{'plot_name': 'Ref. Evapotransp. (FAO)', 'unit': 'mm', 'group': 'climate', 'processing': 'lag_roll'},
    'DWD_soil_moist':{'plot_name': 'Soil Moisture (DWD)', 'unit': '% PAW', 'group': 'climate', 'processing': 'lag_roll'},
    'DWD_soil_temp5cm':{'plot_name': 'Soil Temp 5cm (DWD)', 'unit': '°C', 'group': 'climate', 'processing': 'lag_roll'},
    'ERA5_sm':      {'plot_name': 'Snowmelt (ERA5)', 'unit': 'm', 'group': 'climate', 'processing': 'lag_roll'},
    'ERA5_sf':      {'plot_name': 'Snowfall (ERA5)', 'unit': 'm', 'group': 'climate', 'processing': 'lag_roll'},
    'ERA5_sdwe':    {'plot_name': 'Snow depth (ERA5)', 'unit': 'm', 'group': 'climate', 'processing': 'lag_roll'},
    'ERA5_ssro':    {'plot_name': 'Sub-surface runoff (ERA5)', 'unit': 'm', 'group': 'climate', 'processing': 'lag_roll'},
    'ERA5_sro':     {'plot_name': 'Runoff (ERA5)', 'unit': 'm', 'group': 'climate', 'processing': 'lag_roll'},

    'GWN1000_GR':   {'plot_name': 'Annual GW Recharge', 'unit': 'mm', 'group': 'hydrogeology', 'processing': 'passthrough_num'},
    'HUEK250_HU':   {'plot_name': 'Hydrogeo Unit (HUEK250)', 'unit': 'class', 'group': 'hydrogeology', 'processing': 'passthrough_cat'},
    'HUEK250_K':    {'plot_name': 'Hydraulic Conductivity', 'unit': 'm s-1', 'group': 'hydrogeology', 'processing': 'passthrough_num'},
    'HUEK250_RT':   {'plot_name': 'Rock Type (HUEK250)', 'unit': 'class', 'group': 'hydrogeology', 'processing': 'passthrough_cat'},
    'HUEK250_CT':   {'plot_name': 'Porosity Type (HUEK250)', 'unit': 'class', 'group': 'hydrogeology', 'processing': 'passthrough_cat'},
    'HUEK250_DC':   {'plot_name': 'Consolidation (HUEK250)', 'unit': 'class', 'group': 'hydrogeology', 'processing': 'passthrough_cat'},
    'HUEK250_GC':   {'plot_name': 'Geochem Rock (HUEK250)', 'unit': 'class', 'group': 'hydrogeology', 'processing': 'passthrough_cat'},
    'HYRAUM_HD':    {'plot_name': 'Hydrogeo District', 'unit': 'class', 'group': 'hydrogeology', 'processing': 'passthrough_cat'},
    'HYRAUM_MHD':   {'plot_name': 'Major Hydrogeo Dist.', 'unit': 'class', 'group': 'hydrogeology', 'processing': 'passthrough_cat'},
    'HYSOG_SG':     {'plot_name': 'Hydrologic Soil Group', 'unit': 'class', 'group': 'hydrogeology', 'processing': 'passthrough_cat'},
    'SWR_PR':       {'plot_name': 'Annual Percolation', 'unit': 'mm', 'group': 'hydrogeology', 'processing': 'passthrough_num'},

    'EU_MOHP_DSD_1': {'plot_name': 'Divide-Stream Dist (s1)', 'unit': 'm', 'group': 'hydrology', 'processing': 'passthrough_num'},
    'EU_MOHP_DSD_2': {'plot_name': 'Divide-Stream Dist (s2)', 'unit': 'm', 'group': 'hydrology', 'processing': 'passthrough_num'},
    'EU_MOHP_DSD_3': {'plot_name': 'Divide-Stream Dist (s3)', 'unit': 'm', 'group': 'hydrology', 'processing': 'passthrough_num'},
    'EU_MOHP_SD_1':  {'plot_name': 'Stream Distance (s1)', 'unit': 'm', 'group': 'hydrology', 'processing': 'passthrough_num'},
    'EU_MOHP_SD_2':  {'plot_name': 'Stream Distance (s2)', 'unit': 'm', 'group': 'hydrology', 'processing': 'passthrough_num'},
    'EU_MOHP_SD_3':  {'plot_name': 'Stream Distance (s3)', 'unit': 'm', 'group': 'hydrology', 'processing': 'passthrough_num'},
    'EU_MOHP_LP_1':  {'plot_name': 'Lateral Position (s1)', 'unit': '-', 'group': 'hydrology', 'processing': 'passthrough_num'},
    'EU_MOHP_LP_2':  {'plot_name': 'Lateral Position (s2)', 'unit': '-', 'group': 'hydrology', 'processing': 'passthrough_num'},
    'EU_MOHP_LP_3':  {'plot_name': 'Lateral Position (s3)', 'unit': '-', 'group': 'hydrology', 'processing': 'passthrough_num'},

    'BUEK1000_RSA': {'plot_name': 'Ref Soil Association', 'unit': 'class', 'group': 'lithology', 'processing': 'passthrough_cat'},
    'HUMUS1000_OC': {'plot_name': 'Organic Matter Topsoil', 'unit': 'class', 'group': 'lithology', 'processing': 'passthrough_cat'},

    'CLC_90':       {'plot_name': 'CORINE 1990', 'unit': 'class', 'group': 'landuse', 'processing': 'passthrough_cat'},
    'CLC_00':       {'plot_name': 'CORINE 2000', 'unit': 'class', 'group': 'landuse', 'processing': 'passthrough_cat'},
    'CLC_06':       {'plot_name': 'CORINE 2006', 'unit': 'class', 'group': 'landuse', 'processing': 'passthrough_cat'},
    'CLC_12':       {'plot_name': 'CORINE 2012', 'unit': 'class', 'group': 'landuse', 'processing': 'passthrough_cat'},
    'CLC_18':       {'plot_name': 'CORINE 2018', 'unit': 'class', 'group': 'landuse', 'processing': 'passthrough_cat'},
    'MUNDIALIS_LU': {'plot_name': 'Land Cover 2020', 'unit': 'class', 'group': 'landuse', 'processing': 'passthrough_cat'},

    'GMK1000_GU':   {'plot_name': 'Geomorphic Unit', 'unit': 'class', 'group': 'geomorphology', 'processing': 'passthrough_cat'},
    'DTM20_SL':     {'plot_name': 'Slope', 'unit': '°', 'group': 'geomorphology', 'processing': 'passthrough_num'},
    'DTM20_AS':     {'plot_name': 'Aspect', 'unit': '°', 'group': 'geomorphology', 'processing': 'passthrough_num'},
    'DTM20_GC':     {'plot_name': 'Gen Curvature', 'unit': '-', 'group': 'geomorphology', 'processing': 'passthrough_num'},
    'DTM20_PLC':    {'plot_name': 'Plan Curvature', 'unit': '-', 'group': 'geomorphology', 'processing': 'passthrough_num'},
    'DTM20_PRC':    {'plot_name': 'Profile Curvature', 'unit': '-', 'group': 'geomorphology', 'processing': 'passthrough_num'},
    'DTM20_FA':     {'plot_name': 'Flow Accumulation', 'unit': 'n cells', 'group': 'hydrology', 'processing': 'passthrough_num'},
    'DTM20_TRI':    {'plot_name': 'Terrain Ruggedness', 'unit': '-', 'group': 'hydrology', 'processing': 'passthrough_num'},
    'DTM20_CI':     {'plot_name': 'Convergence/Divergence', 'unit': '-', 'group': 'geomorphology', 'processing': 'passthrough_num'},
    'DTM20_MRI':    {'plot_name': 'Melton Ruggedness', 'unit': '-', 'group': 'hydrology', 'processing': 'passthrough_num'},
    'DTM20_FD':     {'plot_name': 'Flow Direction', 'unit': 'class', 'group': 'geomorphology', 'processing': 'passthrough_cat'},
    'DTM20_TPI':    {'plot_name': 'Topographic Position Index', 'unit': '-', 'group': 'geomorphology', 'processing': 'passthrough_num'},

    'AquiferMed':   {'plot_name': 'Aquifer Medium', 'unit': 'class', 'group': 'hydrogeology', 'processing': 'passthrough_cat'},
    'PreState':     {'plot_name': 'Pressure Condition', 'unit': 'class', 'group': 'hydrogeology', 'processing': 'passthrough_cat'},
    'Elevation':    {'plot_name': 'Surface Elevation', 'unit': 'm', 'group': 'static', 'processing': 'passthrough_num'},
}

# GEMS-GER coverage: dynamic drivers monthly until 2022-12 (no TWSA).
GEMS_DYN_TIME = ('monthly', '1991-01', '2022-12')


def gems_row(col, ftype):
    meta = GEMS_METADATA.get(col, {})
    proc_raw = meta.get('processing', 'lag_roll' if ftype == 'dynamic' else 'passthrough_num')
    # split processing and categorical (passthrough_cat/_num -> passthrough + bool)
    is_cat = proc_raw.endswith('_cat')
    proc = 'lag_roll' if proc_raw == 'lag_roll' else 'passthrough'
    if ftype == 'dynamic':
        temporal, t0, t1 = GEMS_DYN_TIME
    else:
        temporal, t0, t1 = ('static', '', '')
    return {
        'feature_raw_name': col,
        'source_system': 'gems',
        'type': ftype,
        'group': meta.get('group', 'gems_' + ftype),
        'var_group': meta.get('group', 'gems_' + ftype),
        'processing': proc,
        'categorical': is_cat,
        'temporal': temporal,
        'time_start': t0,
        'time_end': t1,
        'plot_name': meta.get('plot_name', col),
        'unit': meta.get('unit', '?'),
    }


features_gems = []
DROP_GEMS_META = {'time', 'GWL', 'time_days', 'MW_ID', 'Proj_ID', 'Operator',
                  'Easting (EPSG:3035)', 'Northing (EPSG:3035)',
                  'Depth', 'UpFilter', 'LoFilter', 'ScrLength',
                  'Unnamed: 0'}   # index column written into the static CSV

print("\nReading GEMS-GER dynamic features...")
for col in get_columns_from_dir(pth_gems_dyn_store):
    if col in DROP_GEMS_META:
        continue
    features_gems.append(gems_row(col, 'dynamic'))

print("Reading GEMS-GER static features...")
for col in get_columns_from_file(pth_gems_stat_store):
    if col in DROP_GEMS_META:
        continue
    features_gems.append(gems_row(col, 'static'))

df_gems = (pd.DataFrame(features_gems)
           .drop_duplicates(subset=['feature_raw_name'])
           .reset_index(drop=True))
df_gems.to_csv(output_meta_gems, index=False, sep=';')
print(f"-> GEMS-GER metadata table saved ({len(df_gems)} features): {output_meta_gems}")
