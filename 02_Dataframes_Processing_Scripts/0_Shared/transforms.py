# -*- coding: utf-8 -*-
"""
transforms.py — named feature transforms referenced by the manifest
===================================================================

These are the non-generic feature builders (manifest `transform=...`). They are
kept here, separate from the generic point-extraction path, so the simple
datasets stay simple and the engineered ones are explicit and unit-testable.

Location: 02_Dataframes_Processing_Scripts/0_Shared/transforms.py
          (imported by 2_Feature_Stores/02_build_dynamic_store.py)

lcc_features : builds 5 engineered land cover features per well from the
               yearly C3S land cover class (LCC, LCC_previous, LCC_changed,
               LCC_stable, months_since_LCC_change). Years before the LCC
               record are back-filled. The LCC NoData class
               (lccs_class == 0) is masked to NaN upstream by the extractor
               via the declared fill value, so it is never used as a class.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# C3S LCC record bounds (years). Stable flag is evaluated only inside this span.
LCC_YEAR_MIN = 1992
LCC_YEAR_MAX = 2022
NO_CHANGE_SENTINEL = -1


def lcc_features(annual: pd.Series, months: pd.DatetimeIndex, *,
                 year_min_lcc: int = LCC_YEAR_MIN,
                 year_max_lcc: int = LCC_YEAR_MAX,
                 back_fill_pre: bool = True,
                 forward_fill_post: bool = False,
                 no_change_sentinel: int = NO_CHANGE_SENTINEL) -> pd.DataFrame:
    """
    Build the five LCC features for ONE well on its monthly index.

    annual : pd.Series indexed by year (int) -> land-cover class for this well
             over the LCC record (NaN where NoData / outside record).
    months : the well's monthly DatetimeIndex (month-start timestamps).

    Returns a DataFrame indexed by `months` with columns:
      LCC, LCC_previous, LCC_changed, months_since_LCC_change, LCC_stable.
    """
    years = pd.Index(months.year)
    w_min, w_max = int(years.min()), int(years.max())

    full_idx = pd.RangeIndex(min(w_min, year_min_lcc),
                             max(w_max, year_max_lcc) + 1, name="year")

    a = annual.reindex(full_idx)
    if back_fill_pre:
        a = a.bfill()
    if forward_fill_post:
        a = a.ffill()

    a_prev = a.shift(1)
    if back_fill_pre:
        a_prev = a_prev.bfill()

    changed = ((a != a_prev) & a.notna() & a_prev.notna()).astype("uint8")

    # most recent change year up to and including each year
    chg_year = np.where(changed.values.astype(bool),
                        full_idx.values.astype("float64"), np.nan)
    last_change = pd.Series(chg_year, index=full_idx).ffill()

    # stable only counts changes inside the real LCC record
    stable = np.uint8(int(changed.loc[year_min_lcc:year_max_lcc].sum()) == 0)

    ymap = pd.Series(years.values, index=months)

    out = pd.DataFrame(index=months)
    out["LCC"]          = ymap.map(a).round().astype("Int16")
    out["LCC_previous"] = ymap.map(a_prev).round().astype("Int16")
    out["LCC_changed"]  = ymap.map(changed).fillna(0).astype("uint8")
    out["LCC_stable"]   = stable

    lcy = ymap.map(last_change).values                 # float w/ NaN
    msc = (months.year.values - lcy) * 12 + (months.month.values - 1)
    out["months_since_LCC_change"] = (
        pd.Series(msc, index=months).fillna(no_change_sentinel).astype("int32")
    )
    return out
