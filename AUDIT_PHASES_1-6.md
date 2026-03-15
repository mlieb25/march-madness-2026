# Professional Audit: March Madness ML Pipeline (Phases 1–6)

**Scope:** Methodology, reproducibility, and rigor of Phases 1 through 6.  
**Date:** 2025-03-15.

---

## Executive Summary

The pipeline demonstrates **strong design choices** (symmetric pairwise features, time-awareness, calibration, BMA/stacking ensemble, Monte Carlo bracket simulation) and is suitable for a portfolio or internal bracket tool. It is weakened by **pipeline fragmentation** (two different Phase 3/4 implementations), **single-year holdout** evaluation, **non–time-aware hyperparameter search** in the documented Phase 3, and **missing handoffs** between the README flow and Phase 5/6.

**Overall methodology rating: 6.5 / 10.**

---

## Phase 1 — Data Engineering

### 1.1 Data pull (`data-pull.py`)

| Aspect | Assessment |
|--------|------------|
| **Sources** | Strong: Barttorvik (historical + 2026 adv), FiveThirtyEight, NCAA NET, Massey, WarrenNolan, optional Sports-Reference. |
| **Robustness** | Timeout (30s) and cloudscraper reduce hangs; no retries or backoff. HTML tables (NET, WarrenNolan) are fragile to layout changes. |
| **Schema** | Torvik columns hardcoded by position (e.g. `'0':'rank', '1':'team', ... '44':'adjt'`). No validation that column indices match current Barttorvik CSV layout. |
| **Reproducibility** | No versioning or checksums of scraped files; re-running can change data silently. |
| **Output** | Writes to `data/`; creates directory if missing. |

**Recommendations:** Add retries with backoff; validate Torvik column names or use a header row if the source provides one; document or snapshot expected schema; consider hashing raw CSVs for reproducibility.

### 1.2 ETL (`etl.py`)

| Aspect | Assessment |
|--------|------------|
| **Label source** | Uses FiveThirtyEight `favorite_win_flag` (tournament outcomes). Appropriate and clearly defined. |
| **Leakage** | Duplicate (year, favorite, underdog) dropped. Training built from historical games only; no future info in features. |
| **Feature design** | 12 features: 6 differentials + 6 ratios (adjOE, adjDE, Barthag, SoS, WAB, Tempo). Symmetric construction: inverse rows added with negated diffs, inverted ratios, and flipped label so model does not see “favorite” as a positional bias. **Strong.** |
| **Name normalization** | Two-pass regex-based `normalize_name` with ordered mappings (e.g. North Carolina State before North Carolina). Handles common collisions; docstring references a prior fix. |
| **Joins** | Inner join to Torvik; only matchups with both teams matched in Torvik for that season remain. Unmatched games dropped (acceptable but worth logging count). |
| **Inference** | Top 68 from NCAA NET; 2026 Torvik stats; fallback to 2025 if 2026 missing (documented). All pairwise matchups generated; same 12-feature schema. |

**Recommendations:** Log number of games dropped at each join; add a small test or assertion that symmetric rows satisfy diff/ratio/label invariants (e.g. as in `test_merge.py`).

**Phase 1 rating: 7 / 10** — Solid ETL and feature design; data-pull and schema coupling are the main risks.

---

## Phase 2 — Baseline (Logistic Regression)

**Implemented in:** `models.py` → `run_phase2_baseline`.

| Aspect | Assessment |
|--------|------------|
| **Split** | Time-aware: train = years &lt; 2014, test = 2014. No shuffle; no cross-year leakage. **Correct.** |
| **Scale** | `StandardScaler` fit on train, transform on test and inference. Appropriate. |
| **Model** | `LogisticRegression(max_iter=1000, random_state=42)`. No regularization specified (default C=1.0). |
| **Evaluation** | Log loss, Brier, ROC-AUC on 2014 holdout. |
| **Inference** | Retrain on **full** historical data (all years) for 2026 predictions; scaler refit on full data. Standard and correct. |
| **Artifacts** | Saves `baseline_predictions_2026.csv` and `data/models/baseline_lr.pkl` (scaler + model). |

**Gaps:** Single holdout year (2014) → high variance; no confidence intervals or multi-year holdout. `models.py` does **not** write `phase3_top_models.json`, so any Phase 4 that expects it (e.g. `phase4.py` for XGB params) must use a fallback.

**Phase 2 rating: 6.5 / 10** — Methodology is sound; evaluation is thin (one year).

---

## Phase 3 — Model Search (XGBoost)

**Implemented in:** `models.py` → `run_phase3_xgboost`.

| Aspect | Assessment |
|--------|------------|
| **Split** | Same as Phase 2: train &lt; 2014, test = 2014. **Correct.** |
| **Scaling** | XGBoost trained on **raw** features (no scaler). Acceptable for tree models but inconsistent with Phase 2 (LR scaled). |
| **Search** | `GridSearchCV(..., cv=3, scoring='neg_log_loss')`. Default `cv=3` is **KFold** — random splits of 2011–2013 data, **not** time-ordered. So validation within train is not time-aware; possible optimism if year effects exist. |
| **Final model** | Best estimator refit on **full** training data for 2026 predictions; saved as `xgb_best.pkl`. |
| **Artifacts** | Writes `xgb_predictions_2026.csv` and `data/models/xgb_best.pkl`. Does **not** write `data/phase3_top_models.json`. |

**Critical:** The README lists Phase 4 as `phase4.py`. That script tries to load `phase3_top_models.json` for XGB params and falls back to hardcoded values. So the **documented** Phase 3 → Phase 4 path never persists “best params” for Phase 4. The **extended** pipeline uses `phase3_model_search.py` (Optuna-style search, rolling time CV) and **does** write `phase3_top_models.json` (elastic_net, gp, lightgbm, xgboost), which is then consumed by `phase4_calibration.py`, not `phase4.py`.

**Phase 3 rating: 5 / 10** — Single-year holdout, non–time-aware CV in grid search, and no export of best params for the README’s Phase 4.

---

## Phase 4 — Calibration

Two implementations exist; they serve different pipelines.

### 4.1 `phase4.py` (README flow)

| Aspect | Assessment |
|--------|------------|
| **Split** | Same: train &lt; 2014, test = 2014. |
| **Calibration** | `CalibratedClassifierCV` with `isotonic` and `sigmoid`, `cv=3` on training data. Evaluates uncalibrated vs calibrated on 2014. |
| **Output** | **Only** `data/calibrated_predictions_2026.csv` (Isotonic-calibrated LR). Does **not** produce `phase4_oof_probs.csv`, `phase4_inference_probs.csv`, or `phase4_best_combos.json`. |
| **Downstream** | Phase 5 expects OOF + inference probs and best_combos. So **Phase 5 cannot run** after only `phase4.py`. |

### 4.2 `phase4_calibration.py` (extended pipeline)

| Aspect | Assessment |
|--------|------------|
| **Input** | Reads `phase3_top_models.json` (from `phase3_model_search.py`). |
| **CV** | Rolling time folds: train on all years before `val_year`, predict `val_year`. Produces OOF probs for stacking. **Time-aware; strong.** |
| **Calibrators** | Multiple (e.g. isotonic, beta, etc.); fit on a calibration set (all but last year), evaluate on last year. ECE and sharpness reported. |
| **Output** | Writes `phase4_oof_probs.csv`, `phase4_inference_probs.csv`, `phase4_best_combos.json`, calibration results, and reliability plots. |

**Phase 4 (README) rating: 5 / 10** — Good calibration idea and evaluation, but wrong outputs for Phase 5 and depends on a file Phase 3 (models.py) never creates.  
**Phase 4 (phase4_calibration.py) rating: 8 / 10** — Rolling OOF, proper calibrator evaluation, and correct handoffs to Phase 5.

---

## Phase 5 — Ensemble

**Implemented in:** `phase5_ensemble.py`. Assumes outputs of **`phase4_calibration.py`** (and thus `phase3_model_search.py`).

| Aspect | Assessment |
|--------|------------|
| **BMA** | Weights ∝ exp(-C_BMA × log_loss) on OOF probs; C_BMA=5. Sensible. |
| **Stacking** | Meta logistic regression on OOF base-model probs; trained where all base columns non-NaN. Uses StandardScaler. |
| **Risk-adaptive** | Separate BMA-style weights for “chalk” vs “upset” games (threshold 0.35 on mean prediction); blend by confidence. Thoughtful. |
| **Final blend** | Fixed: 0.4×BMA + 0.4×stack + 0.2×risk_adaptive. Not tuned on holdout. |
| **Kelly** | Bankroll simulation over historical OOF outcomes; cap 5%; reported per model and BMA ensemble. Good for interpretation. |
| **Kaggle** | Builds submission from ensemble probs and team name → TeamID resolution. |
| **Evaluation** | Meta-model log loss reported is **in-sample** on OOF data. No dedicated holdout evaluation of the ensemble. |

**Gaps:** No formal out-of-sample score for the combined ensemble; blend weights are fixed. Optional: report ensemble log loss on the same holdout year(s) used in Phase 4.

**Phase 5 rating: 7 / 10** — Strong design (BMA + stack + risk-adaptive + Kelly); ensemble itself is not validated on a held-out period.

---

## Phase 6 — Tournament Simulation

**Implemented in:** `phase6_simulation.py`.

| Aspect | Assessment |
|--------|------------|
| **Input** | Reads `phase5_ensemble_probs.csv`; builds bidirectional P(team_a beats team_b) lookup; unknown matchup → 0.5. |
| **Seeds** | Prefer Kaggle `MNCAATourneySeeds.csv` for 2026; fallback to synthetic S-curve from NET (top 68 → 4 regions × 16). |
| **Bracket** | Standard: R1 matchups (1v16, 2v15, …), 4 regions, FF (W vs X, Y vs Z), then championship. Correct. |
| **RNG** | `np.random.default_rng(seed=42)` (configurable); reproducible. |
| **Outputs** | Per-team round-reach probabilities (10k sims default), raw simulation sample, upset-path analysis, three bracket strategies (chalk, exploitative, high-variance), pool EV (e.g. 5k scoring sims) with configurable scoring and upset multiplier. |

**Phase 6 rating: 8 / 10** — Clear, correct bracket logic; useful outputs for bracket strategy and pool EV; minor dependency on seed/team name alignment.

---

## Pipeline Consistency and Documentation

| Issue | Severity |
|-------|----------|
| README describes `models.py` → `phase4.py` → `phase5_ensemble.py`, but Phase 5 requires `phase4_calibration.py` outputs and (for full behavior) `phase3_top_models.json` from `phase3_model_search.py`. | **High** — New users following the README cannot run Phase 5/6 without running the extended pipeline. |
| `models.py` never writes `phase3_top_models.json`. | **High** for README flow. |
| `phase4.py` does not write OOF/inference probs or best_combos. | **High** for Phase 5. |
| Two Phase 3’s (simple grid in `models.py` vs. model search in `phase3_model_search.py`) and two Phase 4’s (`phase4.py` vs. `phase4_calibration.py`) are not clearly distinguished in README. | **Medium** — Confusing; should be documented (e.g. “Quick” vs “Full” pipeline). |

---

## Summary Table

| Phase | Rating | Strengths | Main weaknesses |
|-------|--------|-----------|------------------|
| 1 Data | 7/10 | Symmetric features, clear ETL, name normalization | Brittle data-pull; no schema/version checks |
| 2 Baseline | 6.5/10 | Time-aware split; proper refit for inference | Single-year holdout; no phase3_top_models write |
| 3 Search | 5/10 | Time-aware train/test | GridSearchCV not time-based; no param export for phase4.py |
| 4 Calibration | 5–8/10 | phase4_calibration: rolling OOF, ECE, multiple calibrators | phase4.py: wrong outputs; README flow broken |
| 5 Ensemble | 7/10 | BMA + stack + risk-adaptive; Kelly sim | Fixed blend; no ensemble holdout evaluation |
| 6 Simulation | 8/10 | Correct bracket; pool EV; strategies | Depends on seed/name alignment |

---

## Overall Methodology Rating: **6.5 / 10**

**Rationale:** The core methodology (symmetric pairwise features, time-aware splits where used, calibration, BMA/stacking, simulation) is **above average** and shows good ML and bracket-thinking. The rating is pulled down by: (1) **pipeline fragmentation** and README/implementation mismatch, (2) **single-year holdout** and **non–time-aware** hyperparameter search in the documented Phase 3, (3) **no holdout evaluation** of the final ensemble, and (4) **brittle data acquisition** and lack of reproducibility safeguards. Addressing the pipeline documentation and aligning Phase 3/4 outputs would make the project more robust and easier to reproduce; adding multi-year or rolling holdout and ensemble evaluation would strengthen the methodology further.

---

*End of audit.*
