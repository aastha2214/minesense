"""MineSense: passive land mine detection and classification."""

# On macOS, XGBoost and PyTorch each bundle an OpenMP runtime. If PyTorch's
# loads first, XGBoost crashes ("OMP: Error #179"), so load XGBoost first.
try:
    import xgboost  # noqa: F401
except ImportError:  # the demo app does not need XGBoost
    pass
