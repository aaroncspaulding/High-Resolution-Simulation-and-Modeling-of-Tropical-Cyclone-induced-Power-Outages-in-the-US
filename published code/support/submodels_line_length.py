from __future__ import annotations
import mlx.core as mx
import mlx.nn as nn
import numpy as np

def log_feature(values: np.ndarray, *, divide_by_1000: bool=False) -> np.ndarray:
    values = np.nan_to_num(np.asarray(values, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    values = np.clip(values, 0.0, None)
    if divide_by_1000:
        values = values / 1000.0
    return np.log1p(values).astype(np.float32, copy=False)

def build_features(static) -> np.ndarray:
    return np.column_stack([log_feature(static.length_of_roads_2024, divide_by_1000=True), log_feature(static.total_population), log_feature(static.total_households), log_feature(static.tract_area)]).astype(np.float32, copy=False)

class HierarchicalRoadModel(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.intercept = mx.array(6.1072, dtype=mx.float32)
        self.road_coefficient = mx.array(0.896334, dtype=mx.float32)
        self.context_coefficients = mx.array([-0.638548, 4.93172, 0.0], dtype=mx.float32)
        self.road_context_coefficients = mx.array([0.0, 0.0, 0.0], dtype=mx.float32)

    def __call__(self, x: mx.array) -> mx.array:
        road = x[:, 0]
        context = x[:, 1:4]
        baseline = self.intercept + mx.sum(context * self.context_coefficients, axis=1)
        road_slope = self.road_coefficient + mx.sum(context * self.road_context_coefficients, axis=1)
        return nn.softplus(baseline + road_slope * road)
