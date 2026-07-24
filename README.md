# Published Code
[![DOI](https://zenodo.org/badge/1244694101.svg)](https://doi.org/10.5281/zenodo.21522705)

## Overview
![Figure 1. Overview](images/Figure%201.png)



## Conda dependencies
- Python 3.11, NumPy, pandas, GeoPandas, Matplotlib, PyArrow, and MLX 0.32.
- MLX is installed from Conda Forge and supports Apple-silicon and CPU execution on Linux. See the [MLX install guide](https://ml-explore.github.io/mlx/build/html/install.html) for Linux installation details.

## Run Demo
1. Create the conda environment: `conda env create -f 'published code/environment.yml'`
2. Activate it: `conda activate storm-outage-demo`
3. Prepare Data
4. Run Model: `python 'published code/storm_customers_affected_demo.py' --storm-id 2020279N16284`
5. Outputs available in `published code/output/`

## Demo Data Requirements
- Download `large_data_for_demo` from [Google Drive *here*](https://drive.google.com/drive/folders/1moK39nab_J4nKiLsOOubrguQkS9qAvvN?usp=share_link) and place it directly inside `published code/`.
- The bundle contains processed `weather_data`, `eaglei_targets`, `storm_tracks`, and `county_polygons`.
- Custom locations may be supplied with `--weather-data`, `--eaglei-dir`, `--storm-tracks`, `--county-file`, and `--static-data`.

## Original Dataset Sources
![Table 1. Original datasets](images/Table%201.png)
