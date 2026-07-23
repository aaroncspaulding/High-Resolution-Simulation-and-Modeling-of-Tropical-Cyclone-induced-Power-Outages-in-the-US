from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence
import mlx.core as mx
import numpy as np
import pandas as pd
DEMO_DATA_DIR = Path(__file__).resolve().parents[1] / 'large_data_for_demo'
DEFAULT_STORMS_PATH = DEMO_DATA_DIR / 'storm_tracks' / 'relevant_storm_tracks.feather'
DEFAULT_WEATHER_DB = DEMO_DATA_DIR / 'weather_data'

@dataclass(frozen=True)
class TemporalCellOrder:
    h3_index: np.ndarray

    @property
    def num_cells(self) -> int:
        return int(self.h3_index.shape[0])

@dataclass(frozen=True)
class StormMeta:
    storm_id: str
    storm_name: str
    stems: tuple[str, ...]

@dataclass(frozen=True)
class StormBatch:
    storm_id: str
    storm_name: str
    gust_speed: mx.array
    accumulated_precipitation: mx.array
    outages: mx.array
    storm_track_mask: mx.array

@dataclass(frozen=True)
class AggregatedStorm:
    storm_id: str
    storm_name: str
    weather: np.ndarray
    storm_mask_any: np.ndarray

def _to_utc_naive(ts: pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(ts)
    if ts.tzinfo is None:
        return ts
    return ts.tz_convert('UTC').tz_localize(None)

def _storm_hours(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    start_ = _to_utc_naive(start).floor('h')
    end_ = _to_utc_naive(end).ceil('h') + pd.to_timedelta(1, 'd')
    return pd.date_range(start_, end_, freq='h')

def _file_stem(dt: pd.Timestamp) -> str:
    return dt.strftime('%Y_%m_%dT%H_%M_%S')

def _load_hour_arrays(paths: tuple[str, str, str, str]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    (gust_path, precip_path, outage_path, mask_path) = paths
    return (np.load(gust_path, allow_pickle=False), np.load(precip_path, allow_pickle=False), np.load(outage_path, allow_pickle=False), np.load(mask_path, allow_pickle=False))

def _as_numpy(x, dtype=None) -> np.ndarray:
    arr = np.array(x, copy=False)
    return arr.astype(dtype, copy=False) if dtype is not None else arr

def build_reorder_idx(static_h3_index: np.ndarray, temporal_h3_index: np.ndarray) -> np.ndarray:
    def normalize(values: np.ndarray) -> np.ndarray:
        values = np.asarray(values)
        if np.issubdtype(values.dtype, np.integer):
            return values.astype(np.uint64, copy=False)
        return np.fromiter((int(str(value), 16) for value in values), dtype=np.uint64, count=values.size)
    static_ids = normalize(static_h3_index)
    temporal_ids = normalize(temporal_h3_index)
    temporal_lookup = {int(idx): i for (i, idx) in enumerate(temporal_ids)}
    reorder_idx = np.array([temporal_lookup.get(int(idx), -1) for idx in static_ids], dtype=np.int32)
    missing = int(np.sum(reorder_idx < 0))
    if missing:
        raise ValueError(f'Missing {missing} static cells in temporal order mapping.')
    return reorder_idx

def load_temporal_cell_order(weather_db_directory: Path | str=DEFAULT_WEATHER_DB) -> TemporalCellOrder:
    weather_db_directory = Path(weather_db_directory)
    h3_index = pd.read_feather(weather_db_directory / 'h3_cells_cached.feather', columns=['index'])['index'].to_numpy()
    return TemporalCellOrder(h3_index=h3_index)

class TemporalStormLoader:

    def __init__(self, cell_order: TemporalCellOrder, storms_path: Path | str=DEFAULT_STORMS_PATH, weather_db_directory: Path | str=DEFAULT_WEATHER_DB, storm_names: Optional[Sequence[str]]=None, max_workers: Optional[int]=None, output_dtype=mx.float32):
        self.cell_order = cell_order
        self.storms_path = Path(storms_path)
        self.weather_db_directory = Path(weather_db_directory)
        self.gust_dir = self.weather_db_directory / 'gust_speed'
        self.precip_dir = self.weather_db_directory / 'accumulated_precipitation'
        self.outage_dir = self.weather_db_directory / 'outage'
        self.mask_dir = self.weather_db_directory / 'storm_track_mask'
        self.max_workers = max_workers
        self.output_dtype = output_dtype
        selected_names = {name.upper() for name in storm_names} if storm_names else None
        storms_df = pd.read_feather(self.storms_path)[::-1].reset_index(drop=True)
        if selected_names:
            storms_df = storms_df[storms_df['NAME'].str.upper().isin(selected_names)].reset_index(drop=True)
        storms: list[StormMeta] = []
        for (_, row) in storms_df.iterrows():
            datetimes = _storm_hours(row['datetime_min'], row['datetime_max'])
            storms.append(StormMeta(storm_id=str(row['SID']), storm_name=str(row['NAME']), stems=tuple((_file_stem(dt) for dt in datetimes))))
        if not storms:
            raise ValueError('No storms matched selection.')
        self.storms = storms
        self._storm_by_id = {storm.storm_id: storm for storm in storms}

    def _paths_for_stems(self, stems: Sequence[str]) -> list[tuple[str, str, str, str]]:
        return [(str(self.gust_dir / f'{stem}.npy'), str(self.precip_dir / f'{stem}.npy'), str(self.outage_dir / f'{stem}.npy'), str(self.mask_dir / f'{stem}.npy')) for stem in stems]

    def load_storm(self, storm_id: str) -> StormBatch:
        storm = self._storm_by_id.get(storm_id)
        if storm is None:
            raise ValueError(f'Unknown storm_id: {storm_id}')
        paths = self._paths_for_stems(storm.stems)
        workers = len(paths) if self.max_workers is None else max(1, int(self.max_workers))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            loaded = list(pool.map(_load_hour_arrays, paths))
        gust = mx.stack([mx.array(row[0]) for row in loaded], axis=1)
        precip = mx.stack([mx.array(row[1]) for row in loaded], axis=1)
        outage = mx.stack([mx.array(row[2]) for row in loaded], axis=1)
        mask = mx.stack([mx.array(row[3]) for row in loaded], axis=1)
        if self.output_dtype is not None:
            gust = gust.astype(self.output_dtype)
            precip = precip.astype(self.output_dtype)
            outage = outage.astype(self.output_dtype)
        expected_shape = (self.cell_order.num_cells, len(storm.stems))
        if any(array.shape != expected_shape for array in (gust, precip, outage, mask)):
            raise ValueError(f'Hourly data shape mismatch; expected {expected_shape}.')
        return StormBatch(storm_id=storm.storm_id, storm_name=storm.storm_name, gust_speed=gust, accumulated_precipitation=precip, outages=outage, storm_track_mask=mask)

def build_temporal_loader(cell_order: TemporalCellOrder, storm_name: str | None='ISAIAS', storm_names: Optional[Sequence[str]]=None, storms_path: Path | str=DEFAULT_STORMS_PATH, weather_db_directory: Path | str=DEFAULT_WEATHER_DB, max_workers: Optional[int]=None, output_dtype=mx.float32) -> TemporalStormLoader:
    selected = storm_names if storm_names is not None else (storm_name,) if storm_name else None
    return TemporalStormLoader(cell_order=cell_order, storms_path=storms_path, weather_db_directory=weather_db_directory, storm_names=selected, max_workers=max_workers, output_dtype=output_dtype)

def aggregate_storm_batch(storm_batch: StormBatch, reorder_idx: np.ndarray | mx.array) -> AggregatedStorm:
    reorder_idx = mx.array(reorder_idx, dtype=mx.int32)
    gust = mx.take(storm_batch.gust_speed, reorder_idx, axis=0).astype(mx.float32)
    precip = mx.take(storm_batch.accumulated_precipitation, reorder_idx, axis=0).astype(mx.float32)
    track_mask = mx.take(storm_batch.storm_track_mask, reorder_idx, axis=0).astype(mx.int8)
    valid_hour = track_mask == 0
    zero = mx.array(0.0, dtype=mx.float32)
    neg = mx.array(-1000000000.0, dtype=mx.float32)
    valid_f = valid_hour.astype(mx.float32)
    max_gust = mx.max(mx.where(valid_hour, gust, neg), axis=1)
    max_precip = mx.max(mx.where(valid_hour, precip, neg), axis=1)
    cs_precip = mx.cumsum(mx.where(valid_hour, precip, zero), axis=1)
    cs_valid = mx.cumsum(valid_f, axis=1)
    roll3_precip = cs_precip[:, 2:] - mx.pad(cs_precip[:, :-3], ((0, 0), (1, 0)))
    roll3_valid = cs_valid[:, 2:] - mx.pad(cs_valid[:, :-3], ((0, 0), (1, 0))) >= 2.999
    max_precip_3h = mx.max(mx.where(roll3_valid, roll3_precip, neg), axis=1)
    any_valid_hour = mx.sum(valid_f, axis=1) > 0.0
    any_valid_roll3 = mx.sum(roll3_valid.astype(mx.float32), axis=1) > 0.0
    weather = mx.stack([mx.where(any_valid_hour, max_gust, zero), mx.where(any_valid_hour, max_precip, zero), mx.where(any_valid_roll3, max_precip_3h, zero), mx.sum(((gust >= 20.0) & valid_hour).astype(mx.float32), axis=1), mx.sum(((gust >= 25.0) & valid_hour).astype(mx.float32), axis=1), mx.sum(((gust >= 30.0) & valid_hour).astype(mx.float32), axis=1)], axis=1)
    mx.eval(any_valid_hour, weather)
    return AggregatedStorm(storm_id=storm_batch.storm_id, storm_name=storm_batch.storm_name, weather=_as_numpy(weather, dtype=np.float32), storm_mask_any=_as_numpy(any_valid_hour, dtype=bool))
