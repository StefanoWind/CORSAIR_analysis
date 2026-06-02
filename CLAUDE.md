# CORSAIR Analysis

## Project Description
CORSAIR is a remote sensing wind measurement campaign. This repository covers **experimental design** and **data analysis** for lidar-based wind measurements. The core activities are:

- Optimizing dual-Doppler lidar scan geometries and sector selection for best wind reconstruction.
- Processing raw lidar data through standardization and formatting pipelines.
- Running LiSBOA (LiDAR Statistical Barnes Objective Analysis) to produce gridded statistics of line-of-sight (LOS) velocities.
- Performing dual-Doppler reconstruction to recover horizontal wind vectors (U, V, WS, WD) with uncertainty quantification.
- Validating the reconstructed wind fields.

## Scientific Methods
All methods are grounded in published atmospheric science literature. Key references used in the code:

- LiSBOA: Letizia et al., AMT, 14, 2065–2093, 2021, doi:10.5194/amt-14-2065-2021.
- Dual-Doppler reconstruction assumes negligible vertical velocity (w ≈ 0) and independent uniform errors on LOS velocity (σ_rws) and vertical component (σ_w).

## Dual-Doppler Reconstruction Physics
The LOS velocity model (w ≈ 0):

```
rws_k = cos_ele_k * cos_azi_k * U + cos_ele_k * sin_azi_k * V
```

Written as `[rws1, rws2]^T = M * [U, V]^T`, with inverse giving U and V.

Error model: each lidar measures `rws_k + sin_ele_k * W + noise_k`.
- `Var(rws_k) = σ_rws² + sin_ele_k² * σ_w²`
- `Cov(rws1, rws2) = sin_ele1 * sin_ele2 * σ_w²`  (shared true W, independent noise)

Wind speed error: `σ_WS² = (U/WS)² σ_U² + (V/WS)² σ_V² + 2UV/WS² σ_UV`

Wind direction error (WD = 270 − degrees(arctan2(V, U))):
- `∂WD/∂U = +V/WS² * (180/π)`, `∂WD/∂V = −U/WS² * (180/π)`
- `σ_WD = (180/π) * sqrt((V/WS²)² σ_U² + (U/WS²)² σ_V² − 2UV/WS⁴ σ_UV)`

## Repository Structure
- `utils.py` — core functions: geometry, LiSBOA processing, dual-Doppler reconstruction, plotting.
- `dual-Doppler_site_selector.py` — selects optimal lidar deployment sites.
- `select_scan_sector.py` — selects scan sectors for best dual-Doppler coverage.
- `optimize_scan_*.py` — scan parameter optimization scripts.
- `standardize_lidar.py`, `format_lidar.py`, `download.py` — data ingestion pipeline.
- `validate_dual-Doppler_reconstruction.py`, `validate_dual-Doppler_error.py` — validation scripts.
- `dual-Doppler_stats.py`, `six_beam_profiles.py` — analysis scripts.
