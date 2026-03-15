# March Madness ML Framework

A 7-phase machine learning pipeline for predicting the 2026 NCAA tournament, culminating in a Streamlit portfolio dashboard. See `procedure.docx` for full design spec and `results.md` for logged metrics.

## Setup
```bash
pip install -r requirements.txt
```

## Pipeline variants

There are **two** ways to run the pipeline:

| Variant | Use case | Phases | Phase 5/6? |
|--------|-----------|--------|------------|
| **Quick** | Minimal path: one baseline + XGB, single calibrated output | data-pull → etl → models.py → phase4.py | Yes (phase4.py now writes bridge files) |
| **Full** | Multi-model search, rolling CV, full calibration & ensemble | data-pull → etl → phase3_model_search.py → phase4_calibration.py → phase5 → phase6 | Yes |

- **Phase 5** and **Phase 6** work after either path. The Quick path produces a single-model ensemble (LR+isotonic); the Full path uses multiple model families and calibrators.
- **Default recommendation:** Use the **Full** pipeline (or `python3 run_all.py`) for best methodology; use Quick for a fast end-to-end run.

---

## Quick pipeline — run in order

### Phase 1 — Data Engineering
```bash
python3 data-pull.py   # scrape Barttorvik, 538, NCAA NET, Massey, WarrenNolan
python3 etl.py         # clean, join, compute 12 diff/ratio features → data/ml_training_data.csv
```
> **Note:** `etl.py` must be re-run after any change to `normalize_name` to regenerate
> `data/ml_training_data.csv` and `data/ml_inference_data_2026.csv`.

### Phase 2 & 3 — Baseline + Model Search
```bash
python3 models.py
```
- Phase 2: Logistic Regression baseline → `data/baseline_predictions_2026.csv`, `data/models/baseline_lr.pkl`
- Phase 3: XGBoost grid-search (time-based CV) → `data/xgb_predictions_2026.csv`, `data/models/xgb_best.pkl`, `data/phase3_top_models.json`

### Phase 4 — Calibration
```bash
python3 phase4.py      # isotonic + sigmoid calibration; reads phase3_top_models.json for XGB params
```
Also writes `phase4_oof_probs.csv`, `phase4_inference_probs.csv`, `phase4_best_combos.json` so Phase 5 can run.

### Phase 5 — Ensemble
```bash
python3 phase5_ensemble.py   # BMA + stacking; Kelly bankroll sim → data/phase5_ensemble_probs.csv
```

### Phase 6 — Tournament Simulation
```bash
python3 phase6_simulation.py                       # default: 10k sims, ESPN scoring
python3 phase6_simulation.py --sims 100000 --scoring 1,2,4,8,16,32 --upset-multiplier 2.0
```

### Phase 7 — Streamlit Dashboard
```bash
cd app && streamlit run app.py
```

---

## Full pipeline (recommended for best methodology)

Uses time-aware rolling CV, multiple model families (elastic_net, xgboost, lightgbm, gp), and full calibration/OOF for stacking.

```bash
# One-command run (after Phase 1)
python3 run_all.py
```

Or step by step:
```bash
python3 data-pull.py
python3 etl.py
python3 phase3_model_search.py   # Optuna-style search, writes phase3_top_models.json
python3 phase4_calibration.py    # Rolling OOF, multiple calibrators, writes phase4_* for Phase 5
python3 phase5_ensemble.py
python3 phase6_simulation.py
```

See `run_all.py` for exact order and error handling.

---

## Configuration

Evaluation protocol (holdout year, multi-year eval) is centralized in **`config.py`**:
- `TEST_YEAR`: primary holdout year (default 2014)
- `HOLDOUT_YEARS`: list used for optional multi-year reporting in Phase 2/3

---

## Key output files
| File | Description |
|---|---|
| `data/ml_training_data.csv` | Symmetric training set (2011–present) |
| `data/ml_inference_data_2026.csv` | Possible 2026 matchups |
| `data/phase5_ensemble_probs.csv` | Final ensemble win probabilities |
| `data/phase6_team_round_probs.csv` | P(team reaches each round) across sims |
| `data/phase6_brackets.json` | Chalk, exploitative, and contrarian bracket picks |
| `data/models/` | Persisted sklearn/XGBoost models (joblib) |
| `data/pull_manifest.json` | Checksums/timestamps of raw data (reproducibility) |

For complete metric results, see `results.md`.
