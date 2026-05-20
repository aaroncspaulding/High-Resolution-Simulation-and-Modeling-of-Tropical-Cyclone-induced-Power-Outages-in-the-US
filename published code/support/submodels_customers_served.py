from __future__ import annotations
import mlx.core as mx
import mlx.nn as nn
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
    return np.column_stack([log_feature(static.length_of_roads_2024, divide_by_1000=True), log_feature(static.total_population), log_feature(static.total_households), log_feature(static.tract_area), logit_feature(static.svi_overall)]).astype(np.float32, copy=False)

def segment_sum_np(values: np.ndarray, end_idx: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    end_idx = np.asarray(end_idx, dtype=np.int32)
    if values.size == 0 or end_idx.size == 0:
        return np.zeros(end_idx.shape[0], dtype=np.float32)
    csum = np.cumsum(values, axis=0)
    end_vals = csum[end_idx - 1]
    start_vals = np.concatenate([np.zeros((1,), dtype=np.float64), end_vals[:-1]])
    return (end_vals - start_vals).astype(np.float32, copy=False)

class FeatureNorm(nn.Module):

    def __init__(self, mean: np.ndarray, std: np.ndarray):
        super().__init__()
        self.mean = mx.array(np.asarray(mean, dtype=np.float32), dtype=mx.float32)
        self.inv_std = mx.array((1.0 / np.asarray(std, dtype=np.float32)).astype(np.float32), dtype=mx.float32)
        self.freeze(recurse=False)

    def __call__(self, x: mx.array) -> mx.array:
        return (x - self.mean) * self.inv_std

class CustomersServedLinear(nn.Module):

    def __init__(self, input_mean: np.ndarray, input_std: np.ndarray):
        super().__init__()
        self.input_norm = FeatureNorm(input_mean, input_std)
        self.linear = nn.Linear(int(np.asarray(input_mean).shape[0]), 1)

    def __call__(self, x: mx.array) -> mx.array:
        return mx.squeeze(nn.softplus(self.linear(self.input_norm(x))), axis=-1)
