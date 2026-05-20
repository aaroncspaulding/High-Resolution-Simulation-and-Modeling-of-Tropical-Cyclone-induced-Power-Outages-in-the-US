from __future__ import annotations
from pathlib import Path
import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pandas as pd
PROJECT_ROOT = Path(__file__).resolve().parents[2]
REGION_FEATURES_PATH = PROJECT_ROOT / 'data_cache' / 'static_region_features.feather'
VULNERABILITY_FEATURE_NAMES = ('roads_log', 'population_log', 'households_log', 'tract_area_log', 'svi_overall_logit', 'svi_socioeconomic_status_logit', 'svi_household_characteristics_logit', 'svi_racial_and_ethnic_minority_logit', 'svi_housing_type_and_transportation_logit')
TERRAIN_FEATURE_NAMES = ('tree_canopy_cover', 'tree_canopy_height', 'elevation')
REGION_FEATURE_NAMES = ('region_new_england', 'region_gulf', 'region_eastern_coast')

def as_numpy(x, dtype=None) -> np.ndarray:
    arr = np.array(x, copy=False)
    return arr.astype(dtype, copy=False) if dtype is not None else arr

def segment_sum_np(values: np.ndarray, end_idx: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    end_idx = np.asarray(end_idx, dtype=np.int32)
    if values.size == 0 or end_idx.size == 0:
        return np.zeros(end_idx.shape[0], dtype=np.float32)
    csum = np.cumsum(values, axis=0)
    end_vals = csum[end_idx - 1]
    start_vals = np.concatenate([np.zeros((1,), dtype=np.float64), end_vals[:-1]])
    return (end_vals - start_vals).astype(np.float32, copy=False)

def segment_sum_matrix_np(values: np.ndarray, end_idx: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    end_idx = np.asarray(end_idx, dtype=np.int32)
    if values.size == 0 or end_idx.size == 0:
        return np.zeros((end_idx.shape[0],) + values.shape[1:], dtype=np.float32)
    csum = np.cumsum(values, axis=0)
    end_vals = csum[end_idx - 1]
    start_vals = np.concatenate([np.zeros((1,) + end_vals.shape[1:], dtype=np.float64), end_vals[:-1]], axis=0)
    return (end_vals - start_vals).astype(np.float32, copy=False)

def weighted_segment_mean_matrix_np(values: np.ndarray, weights: np.ndarray, end_idx: np.ndarray) -> np.ndarray:
    numer = segment_sum_matrix_np(np.asarray(values, dtype=np.float32) * np.asarray(weights, dtype=np.float32)[:, None], end_idx)
    denom = segment_sum_np(np.asarray(weights, dtype=np.float32), end_idx)[:, None]
    return (numer / np.maximum(denom, 1e-06)).astype(np.float32, copy=False)

def sanitize_feature(values, *, fill_value: float=0.0, clip_min: float | None=None, clip_max: float | None=None) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=fill_value, posinf=fill_value, neginf=fill_value)
    if clip_min is not None or clip_max is not None:
        arr = np.clip(arr, clip_min, clip_max)
    return arr.astype(np.float32, copy=False)

def normalize_canopy_cover(values) -> np.ndarray:
    arr = sanitize_feature(values, fill_value=0.0, clip_min=0.0)
    upper = float(np.nanmax(arr)) if arr.size else 0.0
    if upper > 1.5:
        return (np.clip(arr, 0.0, 100.0) / 100.0).astype(np.float32, copy=False)
    return np.clip(arr, 0.0, 1.0).astype(np.float32, copy=False)

def build_outage_terrain_features(static_data) -> np.ndarray:
    return np.column_stack([normalize_canopy_cover(static_data.tree_canopy_cover_2023), sanitize_feature(static_data.tree_height_mean, fill_value=0.0, clip_min=0.0), sanitize_feature(static_data.elevation, fill_value=0.0)]).astype(np.float32, copy=False)

def load_region_fixed_effects(h3_index: np.ndarray, region_path: Path=REGION_FEATURES_PATH) -> np.ndarray:
    region_df = pd.read_feather(region_path, columns=['index', *REGION_FEATURE_NAMES])
    region_df['index'] = region_df['index'].astype(str)
    aligned = region_df.set_index('index').reindex([str(idx) for idx in np.asarray(h3_index)])
    if aligned.isna().any().any():
        missing = int(aligned.isna().any(axis=1).sum())
        raise ValueError(f'Missing region fixed effects for {missing} cells.')
    return aligned.to_numpy(dtype=np.float32, copy=False)

def inverse_softplus(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    values = np.clip(values, 1e-06, None)
    return np.where(values > 20.0, values, np.log(np.expm1(values))).astype(np.float32, copy=False)

class FeatureNorm(nn.Module):

    def __init__(self, mean: np.ndarray, std: np.ndarray):
        super().__init__()
        std = np.where(np.asarray(std, dtype=np.float32) < 1e-06, 1.0, std).astype(np.float32, copy=False)
        self.mean = mx.array(np.asarray(mean, dtype=np.float32), dtype=mx.float32)
        self.inv_std = mx.array((1.0 / std).astype(np.float32, copy=False), dtype=mx.float32)
        self.freeze(recurse=False)

    def __call__(self, x: mx.array) -> mx.array:
        return (x - self.mean) * self.inv_std

class PercentUndergroundLinear(nn.Module):

    def __init__(self, input_mean: np.ndarray, input_std: np.ndarray):
        super().__init__()
        self.input_norm = FeatureNorm(input_mean, input_std)
        self.linear = nn.Linear(int(np.asarray(input_mean).shape[0]), 1)

    def __call__(self, x: mx.array) -> mx.array:
        return mx.squeeze(mx.sigmoid(self.linear(self.input_norm(x))), axis=-1)

class TractOutageRateLinear(nn.Module):

    def __init__(self, input_mean: np.ndarray, input_std: np.ndarray, prior_rate: float):
        super().__init__()
        self.input_norm = FeatureNorm(input_mean, input_std)
        self.linear = nn.Linear(int(np.asarray(input_mean).shape[0]), 1)
        self.prior_rate = float(prior_rate)
        self.linear.weight = mx.zeros(self.linear.weight.shape, dtype=mx.float32)
        self.linear.bias = mx.zeros(self.linear.bias.shape, dtype=mx.float32)

    def __call__(self, x: mx.array) -> mx.array:
        return self.prior_rate * mx.exp(mx.squeeze(self.linear(self.input_norm(x)), axis=-1))

class StructuralOutageModel(nn.Module):

    def __init__(self, weather_mean: np.ndarray, weather_std: np.ndarray, vulnerability_mean: np.ndarray, vulnerability_std: np.ndarray, terrain_mean: np.ndarray, terrain_std: np.ndarray, *, max_outages_per_km: float):
        super().__init__()
        weather_std = np.where(np.asarray(weather_std, dtype=np.float32) < 0.0001, 1.0, weather_std).astype(np.float32, copy=False)
        vulnerability_std = np.where(np.asarray(vulnerability_std, dtype=np.float32) < 0.0001, 1.0, vulnerability_std).astype(np.float32, copy=False)
        terrain_std = np.where(np.asarray(terrain_std, dtype=np.float32) < 0.0001, 1.0, terrain_std).astype(np.float32, copy=False)
        self.weather_mean = mx.array(np.asarray(weather_mean, dtype=np.float32), dtype=mx.float32)
        self.weather_inv_std = mx.array((1.0 / np.asarray(weather_std, dtype=np.float32)).astype(np.float32), dtype=mx.float32)
        self.vulnerability_mean = mx.array(np.asarray(vulnerability_mean, dtype=np.float32), dtype=mx.float32)
        self.vulnerability_inv_std = mx.array((1.0 / np.asarray(vulnerability_std, dtype=np.float32)).astype(np.float32), dtype=mx.float32)
        self.terrain_mean = mx.array(np.asarray(terrain_mean, dtype=np.float32), dtype=mx.float32)
        self.terrain_inv_std = mx.array((1.0 / np.asarray(terrain_std, dtype=np.float32)).astype(np.float32), dtype=mx.float32)
        self.vulnerability_linear = nn.Linear(len(VULNERABILITY_FEATURE_NAMES), 1)
        self.terrain_linear = nn.Linear(len(TERRAIN_FEATURE_NAMES), 1)
        self.region_weight = mx.zeros((len(REGION_FEATURE_NAMES),), dtype=mx.float32)
        self.raw_weather_weight = mx.array(inverse_softplus(np.array([1.0, 0.05, 0.05, 0.05, 0.05, 0.05], dtype=np.float32)), dtype=mx.float32)
        self.weather_bias = mx.array(0.0, dtype=mx.float32)
        self.intercept = mx.array(-8.0, dtype=mx.float32)
        self.raw_hazard_scale = mx.array(1.0, dtype=mx.float32)
        self.svi_effect = mx.array(0.07186599, dtype=mx.float32)
        self.max_outages_per_km = float(max_outages_per_km)
        self.vulnerability_linear.weight = mx.zeros(self.vulnerability_linear.weight.shape, dtype=mx.float32)
        self.vulnerability_linear.bias = mx.zeros(self.vulnerability_linear.bias.shape, dtype=mx.float32)
        self.terrain_linear.weight = mx.zeros(self.terrain_linear.weight.shape, dtype=mx.float32)
        self.terrain_linear.bias = mx.zeros(self.terrain_linear.bias.shape, dtype=mx.float32)
        self.freeze(recurse=False, keys=['weather_mean', 'weather_inv_std', 'vulnerability_mean', 'vulnerability_inv_std', 'terrain_mean', 'terrain_inv_std'], strict=True)

    def _probability(self, svi_cell: mx.array, vulnerability_features_cell: mx.array, terrain_features_cell: mx.array, region_features_cell: mx.array, weather_features_cell: mx.array) -> mx.array:
        weather_z = (weather_features_cell - self.weather_mean) * self.weather_inv_std
        vulnerability_z = (vulnerability_features_cell - self.vulnerability_mean) * self.vulnerability_inv_std
        terrain_z = (terrain_features_cell - self.terrain_mean) * self.terrain_inv_std
        weather_weight = nn.softplus(self.raw_weather_weight)
        hazard = mx.sum(weather_z * weather_weight, axis=-1) + self.weather_bias
        vulnerability = mx.squeeze(self.vulnerability_linear(vulnerability_z), axis=-1)
        terrain = mx.squeeze(self.terrain_linear(terrain_z), axis=-1)
        region = mx.sum(region_features_cell * self.region_weight, axis=-1)
        logit = self.intercept + nn.softplus(self.raw_hazard_scale) * hazard + self.svi_effect * (svi_cell - 0.5) + vulnerability + terrain + region
        return mx.sigmoid(logit)

    @staticmethod
    def _overhead_line_km(line_length_cell: mx.array, percent_underground_cell: mx.array) -> mx.array:
        total_line_km = mx.clip(line_length_cell / 1000.0, 0.0, None)
        underground_fraction = mx.clip(percent_underground_cell, 0.0, 1.0)
        return mx.maximum(total_line_km * (1.0 - underground_fraction), 0.0)

    def exposure(self, line_length_cell: mx.array, percent_underground_cell: mx.array, svi_cell: mx.array, vulnerability_features_cell: mx.array, terrain_features_cell: mx.array, region_features_cell: mx.array, weather_features_cell: mx.array) -> tuple[mx.array, mx.array]:
        probability = self._probability(svi_cell, vulnerability_features_cell, terrain_features_cell, region_features_cell, weather_features_cell)
        overhead_line_km = self._overhead_line_km(line_length_cell, percent_underground_cell)
        return (probability * overhead_line_km, probability)

class JointOutageModel(nn.Module):

    def __init__(self, *, weather_mean: np.ndarray, weather_std: np.ndarray, vulnerability_mean: np.ndarray, vulnerability_std: np.ndarray, terrain_mean: np.ndarray, terrain_std: np.ndarray, rate_input_mean: np.ndarray, rate_input_std: np.ndarray, max_outages_per_km: float):
        super().__init__()
        self.outage_model = StructuralOutageModel(weather_mean=weather_mean, weather_std=weather_std, vulnerability_mean=vulnerability_mean, vulnerability_std=vulnerability_std, terrain_mean=terrain_mean, terrain_std=terrain_std, max_outages_per_km=max_outages_per_km)
        self.tract_rate = TractOutageRateLinear(rate_input_mean, rate_input_std, max_outages_per_km)

    def cell_predictions(self, line_length_cell: mx.array, percent_underground_cell: mx.array, svi_cell: mx.array, vulnerability_features_cell: mx.array, terrain_features_cell: mx.array, region_features_cell: mx.array, weather_features_cell: mx.array, tract_features_cell: mx.array) -> tuple[mx.array, mx.array, mx.array]:
        (exposure, probability) = self.outage_model.exposure(line_length_cell, percent_underground_cell, svi_cell, vulnerability_features_cell, terrain_features_cell, region_features_cell, weather_features_cell)
        rate = self.tract_rate(tract_features_cell)
        return (exposure * rate, probability, rate)
