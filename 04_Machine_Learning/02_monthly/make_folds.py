# -*- coding: utf-8 -*-
"""
make_folds.py  --  basin folds for the spatial cross-validation

One-off prerequisite for the spatial CV: assigns whole HydroBASINS basins to
K balanced folds and caches the assignment as
    <meta_dir>/well_folds_<fold_dim>.csv   (MW_ID, fold_id)
(like the twsa_valid_wells.csv cache). mlkit.load_well_folds() reads it.

Source is the static feature store (static_features.csv); only the basin
column <fold_dim>__HYBAS_ID is read, nothing is overwritten.

    python make_folds.py                         # fold_dim=basin_lev06, K=5
    python make_folds.py --fold-dim basin_lev07 --n-folds 5

Greedy longest-processing-time bin packing: whole basins (largest first)
are put into the currently smallest fold -> balanced well numbers, no basin
split across folds (no spatial leakage between folds).
"""

import os
import argparse
import numpy as np
import pandas as pd

from resolve_experiments import PROJECT_ROOT, DEFAULT_META_DIR
import mlkit


def pack_groups_into_folds(group_sizes, k):
    order = sorted(group_sizes.items(), key=lambda kv: kv[1], reverse=True)
    fold_load = [0] * k
    g2f = {}
    for group_id, size in order:
        f = int(np.argmin(fold_load))
        fold_load[f] += size
        g2f[group_id] = f
    return fold_load, g2f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fold-dim', default='basin_lev06',
                    help="prefix of the HYBAS column (-> <fold_dim>__HYBAS_ID)")
    ap.add_argument('--n-folds', type=int, default=5)
    ap.add_argument('--project-root', default=PROJECT_ROOT)
    ap.add_argument('--meta-dir', default=DEFAULT_META_DIR)
    ap.add_argument('--store-profile', default='global')
    args = ap.parse_args()

    paths = mlkit._profile_paths(args.project_root)[args.store_profile]
    basin_col = f"{args.fold_dim}__HYBAS_ID"

    print(f"--- loading static table ---\n{paths['stat_file']}")
    stat = pd.read_csv(paths['stat_file'], sep=',')
    idx = paths['stat_index']
    if idx not in stat.columns:
        raise SystemExit(f"ERROR: index column '{idx}' missing in static_features.csv.")
    if basin_col not in stat.columns:
        raise SystemExit(f"ERROR: '{basin_col}' missing. Available basin columns: "
                         f"{[c for c in stat.columns if c.startswith('basin_')]}")
    stat[idx] = stat[idx].astype(str)
    print(f"Loaded: {len(stat)} wells. Folding by '{basin_col}'.")

    valid = stat.dropna(subset=[basin_col]).copy()
    valid['basin_key'] = valid[basin_col].astype('int64', errors='ignore').astype(str)
    basin_sizes = valid['basin_key'].value_counts()
    print(f"{len(basin_sizes)} unique basins, {len(valid)}/{len(stat)} wells with a basin.")

    fold_load, basin_to_fold = pack_groups_into_folds(basin_sizes.to_dict(), args.n_folds)
    print("\nFold balance (wells per fold):")
    for i, c in enumerate(fold_load):
        print(f"  Fold {i}: {c} wells")
    spread = max(fold_load) - min(fold_load)
    print(f"  Spread: {spread} ({100*spread/len(valid):.1f}% of the folded wells)")

    valid['fold_id'] = valid['basin_key'].map(basin_to_fold).astype('Int64')
    out = valid[[idx, 'fold_id']].rename(columns={idx: 'MW_ID'})
    n_drop = len(stat) - len(out)
    if n_drop:
        print(f"\nNote: {n_drop} wells without a basin -> not in the cache "
              f"(skipped in the spatial training).")

    os.makedirs(args.meta_dir, exist_ok=True)
    cache = os.path.join(args.meta_dir, f"well_folds_{args.fold_dim}.csv")
    out.to_csv(cache, index=False)
    print(f"\nSaved:\n >> {cache}")


if __name__ == '__main__':
    main()
