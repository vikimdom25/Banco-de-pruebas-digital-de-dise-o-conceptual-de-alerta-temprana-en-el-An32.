# Bolt's Performance Journal

This journal documents critical learnings regarding performance optimizations in this codebase.

## 2026-02-23 - [Precomputing scikit-learn standard and MinMaxScaler transformations]
**Learning:** Calling `.transform()` on scikit-learn standard or MinMax scalers in high-frequency loops (like the 50Hz telemetry loop in `ModelWorker` in `testbench/modeldriver.py`) introduces massive overhead. This overhead is due to scikit-learn's input validations, array conversions, and feature name warnings. Vectorizing the mathematical formula directly in NumPy (`(x - mean) / scale` and `x * scale + min`) bypasses this entirely and delivers a ~50x speedup while producing mathematically identical results.
**Action:** When working with scikit-learn scalers in performance-critical or high-frequency loops, always precompute the scaling attributes (`mean_`, `scale_`, `min_`) once during initialization and run the equations natively in NumPy.
