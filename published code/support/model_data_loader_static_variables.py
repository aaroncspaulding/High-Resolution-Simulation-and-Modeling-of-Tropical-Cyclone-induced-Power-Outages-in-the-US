from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STATIC_PATH = PROJECT_ROOT / 'data_cache' / 'aggregated_static_variables.feather'

def _build_end_idx(sorted_group_ids: np.ndarray) -> np.ndarray:
    if sorted_group_ids.size == 0:
        return np.zeros((0,), dtype=np.int32)
    breaks = np.flatnonzero(sorted_group_ids[1:] != sorted_group_ids[:-1]) + 1
    return np.concatenate([breaks, np.array([sorted_group_ids.size], dtype=np.int32)]).astype(np.int32, copy=False)

@dataclass(frozen=True)
class StaticVariablesData:
    h3_index: np.ndarray
    county_fips: np.ndarray
    county_tract_fused_tract_end_idx: np.ndarray
    county_tract_fused_county_end_idx: np.ndarray
    total_population: np.ndarray
    total_households: np.ndarray
    length_of_roads_2024: np.ndarray
    svi_overall: np.ndarray
    svi_socioeconomic_status: np.ndarray
    svi_household_characteristics: np.ndarray
    svi_racial_and_ethnic_minority: np.ndarray
    svi_housing_type_and_transportation: np.ndarray
    tract_area: np.ndarray
    tree_canopy_cover_2023: np.ndarray
    tree_height_mean: np.ndarray
    elevation: np.ndarray

def load_static_variables(static_path: Path | str=DEFAULT_STATIC_PATH) -> StaticVariablesData:
    columns = ['index', 'FULL_COUNTY_FIPS', 'county_map', 'tract_map', 'Total Population Scaled', 'TOTAL NUMBER OF HOUSEHOLDS Scaled', 'length_of_roads_2024', 'svi_overall', 'svi_socioeconomic_status', 'svi_household_characteristics', 'svi_racial_and_ethnic_minority', 'svi_housing_type_and_transportation', 'ALAND', 'tree_canopy_cover_2023', 'tree_height_mean', 'elevation']
    df = pd.read_feather(static_path, columns=columns)
    county_map = np.asarray(df['county_map'].fillna(-1).to_numpy(), dtype=np.int64)
    tract_map = np.asarray(df['tract_map'].fillna(-1).to_numpy(), dtype=np.int64)
    fused_perm = np.lexsort((tract_map, county_map))
    fused_county = county_map[fused_perm]
    fused_tract = tract_map[fused_perm]
    county_end_idx = _build_end_idx(fused_county)
    tract_breaks = np.flatnonzero((fused_county[1:] != fused_county[:-1]) | (fused_tract[1:] != fused_tract[:-1])) + 1
    tract_end_idx = np.concatenate([tract_breaks, np.array([fused_tract.size], dtype=np.int32)]).astype(np.int32, copy=False)
    county_start_idx = np.concatenate([np.array([0], dtype=np.int32), county_end_idx[:-1]]).astype(np.int32, copy=False)

    def take(column: str, *, fill_value: float=0.0) -> np.ndarray:
        values = pd.to_numeric(df[column], errors='coerce').fillna(fill_value).to_numpy(dtype=np.float32, copy=False)
        return values[fused_perm].astype(np.float32, copy=False)
    county_fips_fused = df['FULL_COUNTY_FIPS'].astype(str).str.zfill(5).to_numpy()[fused_perm]
    return StaticVariablesData(h3_index=df['index'].to_numpy()[fused_perm], county_fips=county_fips_fused[county_start_idx], county_tract_fused_tract_end_idx=tract_end_idx, county_tract_fused_county_end_idx=county_end_idx, total_population=take('Total Population Scaled'), total_households=take('TOTAL NUMBER OF HOUSEHOLDS Scaled'), length_of_roads_2024=take('length_of_roads_2024'), svi_overall=take('svi_overall', fill_value=np.nan), svi_socioeconomic_status=take('svi_socioeconomic_status', fill_value=np.nan), svi_household_characteristics=take('svi_household_characteristics', fill_value=np.nan), svi_racial_and_ethnic_minority=take('svi_racial_and_ethnic_minority', fill_value=np.nan), svi_housing_type_and_transportation=take('svi_housing_type_and_transportation', fill_value=np.nan), tract_area=take('ALAND'), tree_canopy_cover_2023=take('tree_canopy_cover_2023'), tree_height_mean=take('tree_height_mean'), elevation=take('elevation'))
