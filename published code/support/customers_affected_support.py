from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
PROCESSED_EAGLEI_DIR = Path('/Users/aaronspaulding/data/processed_eaglei_db')

def load_peak_targets(county_fips: np.ndarray, storm_ids: tuple[str, ...], processed_dir: Path, target_column: str) -> tuple[np.ndarray, np.ndarray]:
    county_fips = np.asarray(county_fips, dtype=object).astype(str)
    storm_pos = {storm_id: idx for (idx, storm_id) in enumerate(storm_ids)}
    target = np.full((len(storm_ids), len(county_fips)), -1.0, dtype=np.float32)
    for (county_idx, fips) in enumerate(county_fips):
        county_file = Path(processed_dir) / f'county_{fips}.feather'
        if not county_file.exists():
            continue
        try:
            df = pd.read_feather(county_file, columns=['storm_id', target_column])
        except Exception:
            continue
        if df.empty:
            continue
        values = pd.to_numeric(df[target_column], errors='coerce').to_numpy(dtype=np.float32, copy=False)
        for (storm_id, value) in zip(df['storm_id'].astype(str), values):
            pos = storm_pos.get(storm_id)
            if pos is not None and np.isfinite(value):
                target[pos, county_idx] = value
    flat = target.reshape(-1).astype(np.float32, copy=False)
    return (flat, flat >= 0.0)
