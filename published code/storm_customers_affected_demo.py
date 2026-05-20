from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
import mlx.core as mx
import numpy as np
import pandas as pd
REPO_ROOT = Path(__file__).resolve().parent.parent
SUPPORT_DIR = Path(__file__).resolve().parent / 'support'
if str(SUPPORT_DIR) not in sys.path:
    sys.path.insert(0, str(SUPPORT_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))
from christines_data import load_counties
from customers_affected_support import PROCESSED_EAGLEI_DIR, load_peak_targets
from model_data_loader_static_variables import load_static_variables
from model_data_loader_temporal_variables import DEFAULT_STORMS_PATH, aggregate_storm_batch, build_reorder_idx, build_temporal_loader, load_temporal_cell_order
from submodels_line_length import HierarchicalRoadModel, build_features as build_line_length_features
from submodels_customers_served import CustomersServedLinear, build_feature_matrix as build_customers_served_feature_matrix, segment_sum_np
from submodels_outage_model import JointOutageModel, PercentUndergroundLinear, StructuralOutageModel, VULNERABILITY_FEATURE_NAMES, as_numpy, build_outage_terrain_features, load_region_fixed_effects, weighted_segment_mean_matrix_np
from submodels_percent_underground import build_feature_matrix as build_static_feature_matrix
SCRIPT_DIR = Path(__file__).resolve().parent
CHECKPOINT_DIR = SCRIPT_DIR / 'checkpoints'
JOINT_WEIGHTS = CHECKPOINT_DIR / 'customers_affected_joint_model_best.safetensors'
SERVED_WEIGHTS = CHECKPOINT_DIR / 'customers_served_linear.safetensors'
LINE_LENGTH_WEIGHTS = CHECKPOINT_DIR / 'hierarchical_road.safetensors'
PERCENT_UNDERGROUND_WEIGHTS = CHECKPOINT_DIR / 'percent_underground_linear_full.safetensors'
DEFAULT_STORM_ID = '2020279N16284'
MAX_OUTAGES_PER_KM = 40.0
ZERO_REGION_FIXED_EFFECTS = False
ZERO_TRACT_AREA_FEATURE = True
TRACT_AREA_FEATURE_IDX = 3
EVAL_BATCH_SIZE = 262144
EPS = 1e-06

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run live outage and customers-affected inference for a storm using repo static data and weather data.')
    parser.add_argument('--storm-id', default=DEFAULT_STORM_ID)
    parser.add_argument('--output', type=Path, default=None)
    return parser.parse_args()

def load_models(served_feature_count: int, rate_feature_count: int) -> tuple[CustomersServedLinear, JointOutageModel]:
    served_model = CustomersServedLinear(np.zeros(served_feature_count, dtype=np.float32), np.ones(served_feature_count, dtype=np.float32))
    served_model.load_weights(str(SERVED_WEIGHTS))
    joint_model = JointOutageModel(weather_mean=np.zeros(6, dtype=np.float32), weather_std=np.ones(6, dtype=np.float32), vulnerability_mean=np.zeros(9, dtype=np.float32), vulnerability_std=np.ones(9, dtype=np.float32), terrain_mean=np.zeros(3, dtype=np.float32), terrain_std=np.ones(3, dtype=np.float32), rate_input_mean=np.zeros(rate_feature_count, dtype=np.float32), rate_input_std=np.ones(rate_feature_count, dtype=np.float32), max_outages_per_km=MAX_OUTAGES_PER_KM)
    joint_model.log_county_outage_scale = mx.array(np.float32(0.0), dtype=mx.float32)
    joint_model.log_storm_peak_slope = mx.array(np.float32(0.0), dtype=mx.float32)
    joint_model.load_weights(str(JOINT_WEIGHTS), strict=False)
    mx.eval(served_model.state, joint_model.state)
    return (served_model, joint_model)

def load_packaged_line_length(static_data) -> np.ndarray:
    model = HierarchicalRoadModel()
    model.load_weights(str(LINE_LENGTH_WEIGHTS))
    pred = model(mx.array(build_line_length_features(static_data), dtype=mx.float32))
    mx.eval(pred, model.state)
    return np.clip(as_numpy(pred, dtype=np.float32), 0.0, None).astype(np.float32, copy=False)

def load_packaged_percent_underground(static_feature_matrix: np.ndarray) -> np.ndarray:
    input_mean = static_feature_matrix.mean(axis=0).astype(np.float32, copy=False)
    input_std = static_feature_matrix.std(axis=0).astype(np.float32, copy=False)
    model = PercentUndergroundLinear(input_mean=input_mean, input_std=input_std)
    model.load_weights(str(PERCENT_UNDERGROUND_WEIGHTS))
    pred = model(mx.array(static_feature_matrix, dtype=mx.float32))
    mx.eval(pred, model.state)
    return np.clip(as_numpy(pred, dtype=np.float32), 0.0, 1.0).astype(np.float32, copy=False)

def load_storm_context(storm_id: str) -> dict[str, object]:
    static_data = load_static_variables()
    static_features = np.asarray(build_static_feature_matrix(static_data), dtype=np.float32)
    vulnerability = static_features[:, :len(VULNERABILITY_FEATURE_NAMES)].astype(np.float32, copy=False)
    terrain = build_outage_terrain_features(static_data)
    region = load_region_fixed_effects(static_data.h3_index)
    line_length = load_packaged_line_length(static_data)
    percent_underground = load_packaged_percent_underground(static_features)
    tract_end_idx = np.asarray(static_data.county_tract_fused_tract_end_idx, dtype=np.int32)
    county_end_idx = np.asarray(static_data.county_tract_fused_county_end_idx, dtype=np.int32)
    county_sizes = np.diff(np.concatenate([np.array([0], dtype=np.int32), county_end_idx])).astype(np.int32, copy=False)
    tract_features = weighted_segment_mean_matrix_np(static_features, np.clip(line_length, 1e-06, None).astype(np.float32, copy=False), tract_end_idx)
    served_features = build_customers_served_feature_matrix(static_data).astype(np.float32, copy=False)
    cell_order = load_temporal_cell_order()
    loader = build_temporal_loader(cell_order=cell_order, storm_name=None, output_dtype=mx.float32)
    reorder_idx = build_reorder_idx(static_data.h3_index, cell_order.h3_index)
    storm_meta = next((storm for storm in loader.storms if storm.storm_id == storm_id), None)
    if storm_meta is None:
        raise ValueError(f'Unknown storm_id: {storm_id}')
    agg = aggregate_storm_batch(loader.load_storm(storm_id), reorder_idx)
    valid_positions = np.flatnonzero(agg.outage_mask_any)
    tract_ids = np.searchsorted(tract_end_idx, valid_positions, side='right').astype(np.int32, copy=False)
    county_ids = np.searchsorted(county_end_idx, valid_positions, side='right').astype(np.int32, copy=False)
    county_coverage = np.bincount(county_ids, minlength=county_end_idx.shape[0]).astype(np.int32, copy=False)
    county_complete_full = (county_coverage == county_sizes).astype(bool, copy=False)
    observed_county_ids = np.unique(county_ids).astype(np.int32, copy=False)
    county_fips_full = pd.Series(np.asarray(static_data.county_fips, dtype=object)).astype(str).str.zfill(5).to_numpy()
    county_order = county_fips_full[observed_county_ids]
    (actual_peak, target_mask_raw) = load_peak_targets(county_order, (storm_id,), PROCESSED_EAGLEI_DIR, 'peak_customers_affected')
    return {'storm_id': storm_id, 'storm_name': str(storm_meta.storm_name), 'served_features': served_features, 'county_end_idx': county_end_idx, 'line_length_cell': line_length[valid_positions].astype(np.float32, copy=False), 'percent_underground_cell': percent_underground[valid_positions].astype(np.float32, copy=False), 'svi_cell': np.asarray(static_data.svi_overall, dtype=np.float32)[valid_positions].astype(np.float32, copy=False), 'vulnerability_cell': vulnerability[valid_positions].astype(np.float32, copy=False), 'terrain_cell': terrain[valid_positions].astype(np.float32, copy=False), 'region_cell': region[valid_positions].astype(np.float32, copy=False), 'weather_cell': agg.weather[valid_positions].astype(np.float32, copy=False), 'tract_features_cell': tract_features[tract_ids].astype(np.float32, copy=False), 'county_ids_full': county_ids, 'observed_county_ids': observed_county_ids, 'county_order': county_order, 'county_complete': county_complete_full[observed_county_ids].astype(bool, copy=False), 'actual_peak': actual_peak.astype(np.float32, copy=False), 'target_mask_raw': target_mask_raw.astype(bool, copy=False)}

def build_county_frame(context: dict[str, object], served_model: CustomersServedLinear, joint_model: JointOutageModel) -> pd.DataFrame:
    pred_served_cell = served_model(mx.array(context['served_features'], dtype=mx.float32))
    mx.eval(pred_served_cell)
    pred_served_full = segment_sum_np(as_numpy(pred_served_cell, dtype=np.float32), np.asarray(context['county_end_idx'], dtype=np.int32))
    pred_customers_served = pred_served_full[np.asarray(context['observed_county_ids'], dtype=np.int32)].astype(np.float32, copy=False)
    vulnerability = np.asarray(context['vulnerability_cell'], dtype=np.float32)
    if ZERO_TRACT_AREA_FEATURE:
        vulnerability = vulnerability.copy()
        vulnerability[:, TRACT_AREA_FEATURE_IDX] = float(as_numpy(joint_model.outage_model.vulnerability_mean[TRACT_AREA_FEATURE_IDX], dtype=np.float32))
    region = np.zeros_like(context['region_cell'], dtype=np.float32) if ZERO_REGION_FIXED_EFFECTS else np.asarray(context['region_cell'], dtype=np.float32)
    pred_cell_parts: list[np.ndarray] = []
    pred_max_parts: list[np.ndarray] = []
    total_rows = int(np.asarray(context['county_ids_full']).shape[0])
    for start in range(0, total_rows, EVAL_BATCH_SIZE):
        end = min(start + EVAL_BATCH_SIZE, total_rows)
        (pred_cell, _pred_prob, pred_rate) = joint_model.cell_predictions(mx.array(np.asarray(context['line_length_cell'])[start:end], dtype=mx.float32), mx.array(np.asarray(context['percent_underground_cell'])[start:end], dtype=mx.float32), mx.array(np.asarray(context['svi_cell'])[start:end], dtype=mx.float32), mx.array(vulnerability[start:end], dtype=mx.float32), mx.array(np.asarray(context['terrain_cell'])[start:end], dtype=mx.float32), mx.array(region[start:end], dtype=mx.float32), mx.array(np.asarray(context['weather_cell'])[start:end], dtype=mx.float32), mx.array(np.asarray(context['tract_features_cell'])[start:end], dtype=mx.float32))
        pred_max = StructuralOutageModel._overhead_line_km(mx.array(np.asarray(context['line_length_cell'])[start:end], dtype=mx.float32), mx.array(np.asarray(context['percent_underground_cell'])[start:end], dtype=mx.float32)) * pred_rate
        mx.eval(pred_cell, pred_max)
        pred_cell_parts.append(as_numpy(pred_cell, dtype=np.float32))
        pred_max_parts.append(as_numpy(pred_max, dtype=np.float32))
    pred_cell = np.concatenate(pred_cell_parts, axis=0).astype(np.float32, copy=False)
    pred_max = np.concatenate(pred_max_parts, axis=0).astype(np.float32, copy=False)
    county_ids_full = np.asarray(context['county_ids_full'], dtype=np.int32)
    full_count = int(np.asarray(context['county_end_idx']).shape[0])
    pred_county_raw_full = np.bincount(county_ids_full, weights=pred_cell.astype(np.float64), minlength=full_count).astype(np.float32, copy=False)
    pred_county_max_full = np.bincount(county_ids_full, weights=pred_max.astype(np.float64), minlength=full_count).astype(np.float32, copy=False)
    observed_county_ids = np.asarray(context['observed_county_ids'], dtype=np.int32)
    county_scale = float(np.exp(as_numpy(joint_model.log_county_outage_scale, dtype=np.float32)))
    pred_county = (county_scale * pred_county_raw_full[observed_county_ids]).astype(np.float32, copy=False)
    pred_county_max = pred_county_max_full[observed_county_ids].astype(np.float32, copy=False)
    pred_county_pct = np.clip(pred_county / np.maximum(pred_county_max, EPS), 0.0, 1.0).astype(np.float32, copy=False)
    actual_peak = np.asarray(context['actual_peak'], dtype=np.float32)
    target_mask_raw = np.asarray(context['target_mask_raw'], dtype=bool)
    county_complete = np.asarray(context['county_complete'], dtype=bool)
    pred_peak = (pred_county_pct * np.clip(pred_customers_served, 0.0, None)).astype(np.float32, copy=False)
    peak_target_mask = target_mask_raw & county_complete & (pred_customers_served > 0.0) & (pred_county_max > EPS)
    return pd.DataFrame({'county_fips': np.asarray(context['county_order'], dtype=object), 'actual_peak_customers_affected': actual_peak, 'pred_peak_customers_affected': pred_peak, 'pred_county_outages': pred_county, 'peak_target_mask': peak_target_mask})

def default_output_path(storm_name: str, storm_id: str) -> Path:
    slug = ''.join((ch.lower() if ch.isalnum() else '_' for ch in storm_name)).strip('_') or storm_id.lower()
    return SCRIPT_DIR / 'output' / f'{slug}_{storm_id}_customers_affected_actual_predicted_and_outages.png'

def plot_storm(county_frame: pd.DataFrame, *, storm_id: str, storm_name: str, output_path: Path) -> dict[str, object]:
    counties = load_counties().copy()
    counties['GEOID'] = counties['GEOID'].astype(str).str.zfill(5)
    counties = counties[counties['GEOID'].isin(set(county_frame['county_fips'].astype(str)))].copy()
    if counties.crs is None:
        counties = counties.set_crs('EPSG:4326', allow_override=True)
    merged = counties.merge(county_frame, left_on='GEOID', right_on='county_fips', how='left')
    merged['peak_target_mask'] = merged['peak_target_mask'].fillna(False).astype(bool)
    target_only = merged[merged['peak_target_mask']].copy()
    if target_only.empty:
        raise ValueError('No evaluation counties remain after masking.')
    track = gpd.read_feather(DEFAULT_STORMS_PATH)
    track = track[track['SID'].astype(str) == storm_id].copy()
    track_geom = track.geometry.iloc[0] if not track.empty else None
    (minx, miny, maxx, maxy) = target_only.total_bounds
    x_pad = max((maxx - minx) * 0.08, 0.75)
    y_pad = max((maxy - miny) * 0.08, 0.55)
    affected_values = np.concatenate([target_only['actual_peak_customers_affected'].to_numpy(dtype=np.float64), target_only['pred_peak_customers_affected'].to_numpy(dtype=np.float64)])
    affected_values = affected_values[np.isfinite(affected_values) & (affected_values > 0.0)]
    affected_norm = PowerNorm(gamma=0.35, vmin=0.0, vmax=max(float(affected_values.max()) if affected_values.size else 1.0, 1.0))
    outage_values = target_only['pred_county_outages'].to_numpy(dtype=np.float64)
    outage_values = outage_values[np.isfinite(outage_values) & (outage_values > 0.0)]
    outage_norm = PowerNorm(gamma=0.35, vmin=0.0, vmax=max(float(outage_values.max()) if outage_values.size else 1.0, 1.0))
    (fig, axes) = plt.subplots(1, 3, figsize=(18.0, 6.8), dpi=260)
    panels = (('actual_peak_customers_affected', affected_norm, 'magma', f"Actual customers affected | total={float(target_only['actual_peak_customers_affected'].sum()):,.0f}"), ('pred_peak_customers_affected', affected_norm, 'magma', f"Predicted customers affected | total={float(target_only['pred_peak_customers_affected'].sum()):,.0f}"), ('pred_county_outages', outage_norm, 'viridis', f"Predicted outages | total={float(target_only['pred_county_outages'].sum()):,.0f}"))
    for (ax, (column, norm, cmap, title)) in zip(axes, panels):
        merged.plot(ax=ax, color='#f5f5f5', edgecolor='#d4d4d8', linewidth=0.2)
        target_only.plot(ax=ax, column=column, cmap=cmap, edgecolor='#52525b', linewidth=0.2, norm=norm)
        if track_geom is not None:
            gpd.GeoSeries([track_geom], crs=merged.crs).plot(ax=ax, color='black', linewidth=1.4, alpha=0.95, zorder=5)
        ax.set_xlim(minx - x_pad, maxx + x_pad)
        ax.set_ylim(miny - y_pad, maxy + y_pad)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_frame_on(False)
        ax.set_title(title, fontsize=11, pad=6)
    affected_bar = plt.cm.ScalarMappable(norm=affected_norm, cmap='magma')
    affected_bar.set_array([])
    outage_bar = plt.cm.ScalarMappable(norm=outage_norm, cmap='viridis')
    outage_bar.set_array([])
    fig.colorbar(affected_bar, ax=axes[:2], orientation='vertical', shrink=0.68, pad=0.015, fraction=0.04).set_label('Customers affected')
    fig.colorbar(outage_bar, ax=[axes[2]], orientation='vertical', shrink=0.68, pad=0.015, fraction=0.06).set_label('Predicted outages')
    fig.suptitle(f'{storm_name} ({storm_id}) county impacts', fontsize=18, y=0.985)
    fig.text(0.5, 0.945, f'evaluation counties={target_only.shape[0]:,} | static + weather driven inference', ha='center', va='top', fontsize=10)
    fig.subplots_adjust(left=0.02, right=0.94, bottom=0.03, top=0.89, wspace=0.03)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches='tight')
    plt.close(fig)
    return {'storm_id': storm_id, 'storm_name': storm_name, 'target_county_count': int(target_only.shape[0]), 'actual_peak_total': float(target_only['actual_peak_customers_affected'].sum()), 'predicted_peak_total': float(target_only['pred_peak_customers_affected'].sum()), 'predicted_outage_total': float(target_only['pred_county_outages'].sum()), 'plot_path': str(output_path.resolve())}

def main() -> None:
    args = parse_args()
    context = load_storm_context(args.storm_id)
    (served_model, joint_model) = load_models(served_feature_count=int(np.asarray(context['served_features']).shape[1]), rate_feature_count=int(np.asarray(context['tract_features_cell']).shape[1]))
    output_path = args.output if args.output is not None else default_output_path(str(context['storm_name']), str(context['storm_id']))
    summary = plot_storm(build_county_frame(context, served_model, joint_model), storm_id=str(context['storm_id']), storm_name=str(context['storm_name']), output_path=output_path)
    summary['raw_target_count'] = int(np.sum(np.asarray(context['target_mask_raw'], dtype=bool)))
    summary['raw_target_actual_peak_total'] = float(np.sum(np.clip(np.asarray(context['actual_peak'], dtype=np.float64)[np.asarray(context['target_mask_raw'], dtype=bool)], 0.0, None)))
    print(json.dumps(summary, indent=2))
if __name__ == '__main__':
    main()
