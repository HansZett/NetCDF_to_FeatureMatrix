# -*- coding: utf-8 -*-
r"""
resolve_experiments.py  --  resolve the experiment configuration

Resolves experiments.yaml into an explicit feature allowlist per experiment
and validates it against the feature metadata tables
(features_meta_global.csv / features_meta_GEMS-GER.csv, written by
02_Dataframes_Processing_Scripts/3_Feature_Metadata).

Every experiment also carries a resolved target block and split block:
    target: {kind: zscore|gwa_cm, reference_start, reference_end, gwl_to_cm, min_ref_months}
    split:  {kind: temporal|spatial, fold_dim, n_folds, val_fold_offset}
Defaults come from global_settings in experiments.yaml (or DEFAULT_TARGET /
DEFAULT_SPLIT below): zscore target + temporal split.

Resolution:
    resolved = ( resolved(base) u expand(include_groups) u include_features )
               \ ( expand(exclude_groups) u exclude_features )

Writes into the metadata folder (03_Dataframes/03_Metadata):
    resolved_allowlists.json     -> per experiment: profile, output folder, hull,
                                    target, split, feature list.
    resolved_features_global.csv / resolved_features_gems.csv (TRUE/FALSE matrix)
    resolved_summary.csv

Loader for the ML scripts:
    from resolve_experiments import load_experiment
    exp = load_experiment("ERA5_4_twsa")
    exp.features, exp.target, exp.split, exp.store_profile, ...

Location: 04_Machine_Learning/02_monthly/
Run     : python resolve_experiments.py   (after every change of experiments.yaml)
"""

import os
import json
import argparse
from dataclasses import dataclass

import pandas as pd
import yaml


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------
# PATHS  (the project root is the first parent folder containing
# 03_Dataframes; it can be overridden with the environment variable
# ML_PROJECT_ROOT)
# ---------------------------------------------------------------------
def find_project_root(start=SCRIPT_DIR, marker="03_Dataframes"):
    d = os.path.abspath(start)
    while True:
        if os.path.isdir(os.path.join(d, marker)):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            return os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
        d = parent


PROJECT_ROOT = os.environ.get("ML_PROJECT_ROOT") or find_project_root()
DEFAULT_META_DIR = os.path.join(PROJECT_ROOT, "03_Dataframes", "03_Metadata")
DEFAULT_CONFIG = os.path.join(SCRIPT_DIR, "experiments.yaml")

META_FILES = {
    'global': 'features_meta_global.csv',
    'gems':   'features_meta_GEMS-GER.csv',
}

# Defaults for the target and split blocks (used if neither global_settings nor
# the experiment set them): zscore target + temporal split.
DEFAULT_TARGET = {
    'kind': 'zscore',
    'reference_start': '2004-01-01',
    'reference_end':   '2009-12-31',
    'gwl_to_cm':       100.0,
    'min_ref_months':  12,
}
DEFAULT_SPLIT = {
    'kind': 'temporal',
    'fold_dim': 'basin_lev06',     # prefix of the HYBAS column (-> <fold_dim>__HYBAS_ID)
    'n_folds': 5,
    'val_fold_offset': 1,
}


class ConfigError(Exception):
    pass


# ---------------------------------------------------------------------
# Feature metadata tables
# ---------------------------------------------------------------------
def load_meta(meta_dir):
    meta = {}
    for profile, fname in META_FILES.items():
        path = os.path.join(meta_dir, fname)
        if not os.path.exists(path):
            print(f"Note: metadata table for profile '{profile}' missing ({path}).")
            continue
        df = pd.read_csv(path, sep=';', dtype=str).fillna('')
        cols = list(df['feature_raw_name'])
        group_map = {}
        for _, r in df.iterrows():
            group_map.setdefault(r['group'], []).append(r['feature_raw_name'])
        time_end = dict(zip(df['feature_raw_name'], df['time_end']))
        meta[profile] = {'columns': set(cols), 'group_map': group_map,
                         'time_end': time_end, 'order': cols}
    return meta


def _as_list(spec, *keys):
    out = []
    for k in keys:
        v = spec.get(k)
        if v:
            if not isinstance(v, list):
                raise ConfigError(f"'{k}' must be a list, got {type(v).__name__}.")
            out.extend(v)
    return out


# ---------------------------------------------------------------------
# Profile / feature resolution
# ---------------------------------------------------------------------
def resolve_profile(name, experiments, _stack=None):
    _stack = _stack or []
    if name in _stack:
        raise ConfigError(f"Cycle in the base_experiment chain: {' -> '.join(_stack + [name])}")
    spec = experiments[name]
    if spec.get('store_profile'):
        return spec['store_profile']
    base = spec.get('base_experiment')
    if base:
        return resolve_profile(base, experiments, _stack + [name])
    return 'global'


def expand_groups(groups, profile, meta, exp_name, kind):
    cols = []
    gmap = meta[profile]['group_map']
    for g in groups:
        if g not in gmap:
            avail = ', '.join(sorted(gmap))
            raise ConfigError(f"[{exp_name}] {kind}: group '{g}' does not exist in "
                              f"profile '{profile}'. Available: {avail}")
        cols.extend(gmap[g])
    return cols


def validate_features(feats, profile, meta, exp_name, kind):
    cols = meta[profile]['columns']
    unknown = [f for f in feats if f not in cols]
    if unknown:
        raise ConfigError(f"[{exp_name}] {kind}: unknown feature(s) in profile "
                          f"'{profile}': {unknown}.")


def resolve_features(name, experiments, meta, cache, _stack=None):
    if name in cache:
        return cache[name]
    _stack = _stack or []
    if name in _stack:
        raise ConfigError(f"Cycle in the base_experiment chain: {' -> '.join(_stack + [name])}")

    spec = experiments[name]
    profile = resolve_profile(name, experiments)
    if profile not in meta:
        raise ConfigError(f"[{name}] store_profile '{profile}' has no loaded "
                          f"metadata table ({META_FILES.get(profile, '?')}).")

    base = spec.get('base_experiment')
    if base:
        if base not in experiments:
            raise ConfigError(f"[{name}] base_experiment '{base}' not defined.")
        if resolve_profile(base, experiments) != profile:
            raise ConfigError(f"[{name}] base_experiment '{base}' has a different "
                              f"store_profile -> inheritance across stores is not allowed.")
        base_set = set(resolve_features(base, experiments, meta, cache, _stack + [name]))
    else:
        base_set = set()

    inc_groups = _as_list(spec, 'include_groups')
    inc_feats  = _as_list(spec, 'include_features', 'features')
    exc_groups = _as_list(spec, 'exclude_groups')
    exc_feats  = _as_list(spec, 'exclude_features')

    validate_features(inc_feats, profile, meta, name, 'include_features')
    validate_features(exc_feats, profile, meta, name, 'exclude_features')

    inc = set(expand_groups(inc_groups, profile, meta, name, 'include_groups')) | set(inc_feats)
    exc = set(expand_groups(exc_groups, profile, meta, name, 'exclude_groups')) | set(exc_feats)
    resolved = (base_set | inc) - exc

    noop = exc - (base_set | inc)
    if noop:
        print(f"  [{name}] Note: exclude has no effect: {sorted(noop)}")
    if not resolved:
        raise ConfigError(f"[{name}] empty feature allowlist after resolution.")

    cache[name] = sorted(resolved, key=lambda c: meta[profile]['order'].index(c))
    return cache[name]


# ---------------------------------------------------------------------
# Hull (time window)
# ---------------------------------------------------------------------
def hull_for(name, experiments, gs, features, profile, meta):
    spec = experiments[name]
    time_start = spec.get('time_start', gs.get('time_start'))
    time_end   = spec.get('time_end',   gs.get('time_end'))
    twsa_valid = spec.get('require_twsa_valid', gs.get('require_twsa_valid', False))
    ends = [meta[profile]['time_end'].get(f, '') for f in features]
    ends = [e for e in ends if e]
    nat_end = min(ends) if ends else ''
    note = ''
    if nat_end and time_end:
        if pd.Period(time_end, 'M') > pd.Period(nat_end, 'M'):
            note = (f"Cap {time_end} > natural end {nat_end}: a used "
                    f"feature is NaN after {nat_end} -> check the cap against {nat_end}.")
        elif pd.Period(time_end, 'M') < pd.Period(nat_end, 'M'):
            note = (f"Cap {time_end} < natural end {nat_end}: {nat_end}"
                    f"->{time_end} deliberately dropped (uniform hull).")
    return {'time_start': time_start, 'time_end': time_end,
            'require_twsa_valid': bool(twsa_valid),
            'natural_time_end': nat_end, 'hull_note': note}


# ---------------------------------------------------------------------
# Resolve target / split (global_settings < experiment)
# ---------------------------------------------------------------------
def target_for(spec, gs):
    return {**DEFAULT_TARGET, **(gs.get('target') or {}), **(spec.get('target') or {})}


def split_for(spec, gs):
    return {**DEFAULT_SPLIT, **(gs.get('split') or {}), **(spec.get('split') or {})}


def _check_spatial_leakage(name, split, features):
    """With a spatial split the fold key (<fold_dim>__HYBAS_ID) must not be a
    feature -- otherwise the model learns the fold membership directly."""
    if split['kind'] != 'spatial':
        return
    hyb = f"{split['fold_dim']}__HYBAS_ID"
    if hyb in features:
        raise ConfigError(
            f"[{name}] spatial split: '{hyb}' is in the allowlist -> leakage. "
            f"Remove it with exclude_features.")


# ---------------------------------------------------------------------
# Main run
# ---------------------------------------------------------------------
def build(config_path, meta_dir, out_dir):
    with open(config_path, encoding='utf-8') as fh:
        cfg = yaml.safe_load(fh)
    gs = cfg.get('global_settings', {})
    experiments = cfg.get('experiments', {})
    if not experiments:
        raise ConfigError("No 'experiments' in the config.")

    meta = load_meta(meta_dir)
    os.makedirs(out_dir, exist_ok=True)

    cache, allowlists, summary_rows = {}, {}, []

    for name in experiments:
        feats = resolve_features(name, experiments, meta, cache)
        profile = resolve_profile(name, experiments)
        spec = experiments[name]
        hull = hull_for(name, experiments, gs, feats, profile, meta)
        target = target_for(spec, gs)
        split = split_for(spec, gs)
        _check_spatial_leakage(name, split, feats)

        out_sub = spec.get('output_subdir', f"results_{name.lower()}")
        results_root = gs.get('results_root', '')
        out_path = os.path.join(results_root, out_sub) if results_root else out_sub

        allowlists[name] = {
            'description': spec.get('description', ''),
            'store_profile': profile,
            'output_subdir': out_sub,
            'output_path': out_path,
            'display_name': spec.get('display_name', name),
            'color': spec.get('color', ''),
            'stop_start': gs.get('stop_start', ''),
            'test_start': gs.get('test_start', ''),
            **hull,
            'target': target,
            'split': split,
            'n_features': len(feats),
            'features': feats,
        }
        summary_rows.append({
            'experiment': name, 'store_profile': profile, 'n_features': len(feats),
            'output_subdir': out_sub, 'time_start': hull['time_start'],
            'time_end': hull['time_end'], 'natural_time_end': hull['natural_time_end'],
            'require_twsa_valid': hull['require_twsa_valid'],
            'target_kind': target['kind'], 'split_kind': split['kind'],
            'fold_dim': split['fold_dim'] if split['kind'] == 'spatial' else '',
            'hull_note': hull['hull_note'],
        })
        extra = f"  target={target['kind']:<7} split={split['kind']}"
        if split['kind'] == 'spatial':
            extra += f"({split['fold_dim']}, K={split['n_folds']})"
        print(f"  {name:<24} profile={profile:<6} n_features={len(feats):>3}{extra}  -> {out_sub}")
        if hull['hull_note']:
            print(f"      hull: {hull['hull_note']}")

    json_path = os.path.join(out_dir, 'resolved_allowlists.json')
    with open(json_path, 'w', encoding='utf-8') as fh:
        json.dump({'global_settings': gs, 'experiments': allowlists},
                  fh, indent=2, ensure_ascii=False)

    for profile in meta:
        exp_names = [n for n in experiments if resolve_profile(n, experiments) == profile]
        if not exp_names:
            continue
        index = meta[profile]['order']
        mat = pd.DataFrame(False, index=index, columns=exp_names)
        for n in exp_names:
            mat.loc[[f for f in cache[n]], n] = True
        mat.index.name = 'feature_raw_name'
        mat_path = os.path.join(out_dir, f'resolved_features_{profile}.csv')
        mat.to_csv(mat_path, sep=';')
        print(f"-> Matrix ({profile}): {mat.shape[0]} x {mat.shape[1]}  {mat_path}")

    sum_path = os.path.join(out_dir, 'resolved_summary.csv')
    pd.DataFrame(summary_rows).to_csv(sum_path, sep=';', index=False)
    print(f"-> Allowlists: {json_path}")
    print(f"-> Summary:    {sum_path}")
    return allowlists


# ---------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------
@dataclass
class ResolvedExperiment:
    name: str
    description: str
    store_profile: str
    output_subdir: str
    output_path: str
    display_name: str
    color: str
    time_start: str
    time_end: str
    stop_start: str
    test_start: str
    require_twsa_valid: bool
    natural_time_end: str
    hull_note: str
    n_features: int
    features: list
    target: dict = None
    split: dict = None


def load_experiment(name, meta_dir=DEFAULT_META_DIR):
    json_path = os.path.join(meta_dir, 'resolved_allowlists.json')
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"{json_path} missing -- run resolve_experiments.py first.")
    with open(json_path, encoding='utf-8') as fh:
        data = json.load(fh)
    if name not in data['experiments']:
        raise KeyError(f"Experiment '{name}' not in {json_path}. "
                       f"Available: {list(data['experiments'])}")
    e = data['experiments'][name]
    base_keys = ('description', 'store_profile', 'output_subdir', 'output_path',
                 'display_name', 'color', 'time_start', 'time_end', 'stop_start',
                 'test_start', 'require_twsa_valid', 'natural_time_end', 'hull_note',
                 'n_features', 'features')
    kwargs = {k: e[k] for k in base_keys}
    kwargs['target'] = e.get('target', dict(DEFAULT_TARGET))
    kwargs['split']  = e.get('split',  dict(DEFAULT_SPLIT))
    return ResolvedExperiment(name=name, **kwargs)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default=DEFAULT_CONFIG)
    p.add_argument('--meta-dir', default=DEFAULT_META_DIR)
    p.add_argument('--out-dir', default=DEFAULT_META_DIR)
    args = p.parse_args()
    build(args.config, args.meta_dir, args.out_dir)


if __name__ == '__main__':
    main()
