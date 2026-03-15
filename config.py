"""
Single source of truth for evaluation protocol across phases.
Used by models.py, phase4.py, phase4_calibration.py, and phase5_ensemble.py.
"""
# Primary holdout year for single-split evaluation (Phase 2, 3, phase4.py)
TEST_YEAR = 2014

# Multiple holdout years for optional multi-year reporting (Phase 2/3)
# Each year is evaluated with train = all years before it
HOLDOUT_YEARS = [2014, 2015, 2016]

# Minimum number of training years before a year can be used as holdout
MIN_TRAIN_YEARS = 2
