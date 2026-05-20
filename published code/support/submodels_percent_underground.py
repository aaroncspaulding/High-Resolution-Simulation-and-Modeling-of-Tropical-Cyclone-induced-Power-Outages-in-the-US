from __future__ import annotations
import numpy as np

def sanitize_feature(values, *, fill_value: float=0.0, clip_min: float | None=None) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=fill_value, posinf=fill_value, neginf=fill_value)
    if clip_min is not None:
        arr = np.clip(arr, clip_min, None)
    return arr.astype(np.float32, copy=False)

def sanitize_svi(values) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.5, posinf=1.0, neginf=0.0)
    arr = np.where(arr < 0.0, 0.5, arr)
    return np.clip(arr, 0.0, 1.0).astype(np.float32, copy=False)

def log_feature(values, *, divide_by_1000: bool=False) -> np.ndarray:
    arr = sanitize_feature(values, clip_min=0.0)
    if divide_by_1000:
        arr = arr / 1000.0
    return np.log1p(arr).astype(np.float32, copy=False)

def logit_feature(values, *, eps: float=0.0001) -> np.ndarray:
    arr = np.clip(sanitize_svi(values), eps, 1.0 - eps)
    return (np.log(arr) - np.log1p(-arr)).astype(np.float32, copy=False)

def build_feature_matrix(static) -> np.ndarray:
    base = np.column_stack([log_feature(static.length_of_roads_2024, divide_by_1000=True), log_feature(static.total_population), log_feature(static.total_households), log_feature(static.tract_area), logit_feature(static.svi_overall), logit_feature(static.svi_socioeconomic_status), logit_feature(static.svi_household_characteristics), logit_feature(static.svi_racial_and_ethnic_minority), logit_feature(static.svi_housing_type_and_transportation)]).astype(np.float32, copy=False)
    tract_area_anchor = base[:, 3:4]
    svi_anchor = base[:, 4:5]
    return np.concatenate([base, base * tract_area_anchor, base * svi_anchor], axis=1).astype(np.float32, copy=False)
