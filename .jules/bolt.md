# Bolt's Performance Journal

## 2026-03-01 - Vectorized Mathematical Scaling in Hot Paths
**Learning:** scikit-learn transformers (like `StandardScaler` and `MinMaxScaler`) introduce significant validation overhead when calling `.transform()` in high-frequency loops (e.g., 50Hz telemetry loop). Replacing these with basic NumPy/mathematical equations using pre-extracted scaler parameters (`mean_`, `scale_`, `min_`) provides an ~130x speedup while yielding mathematically identical results.
**Action:** Extract scaler parameters once during initialization and run vectorized equations directly on incoming NumPy/dict telemetry data in high-frequency/hot-path loops.
