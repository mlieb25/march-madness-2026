# Recommended Fixes (from Audit Phases 1–6)

Actionable list to address issues in `AUDIT_PHASES_1-6.md`. Order is by impact and dependency (pipeline first, then phases).

---

## 1. Pipeline & documentation (high impact)

| # | Fix | Why |
|---|-----|-----|
| 1.1 | **Document two pipelines in README.** Add a "Pipeline variants" section: **(A) Quick:** `data-pull.py` → `etl.py` → `models.py` → `phase4.py` (produces `calibrated_predictions_2026.csv` only; no Phase 5/6). **(B) Full:** `data-pull.py` → `etl.py` → `phase3_model_search.py` → `phase4_calibration.py` → `phase5_ensemble.py` → `phase6_simulation.py`. State that Phase 5 and Phase 6 require the Full pipeline. | Removes confusion and broken expectations. |
| 1.2 | **Make README "Quick" path complete through Phase 6.** Option A: After `phase4.py`, add a small script (or flag) that builds `phase4_oof_probs.csv`, `phase4_inference_probs.csv`, and `phase4_best_combos.json` from the single calibrated LR model so Phase 5 can run. Option B: In README, recommend the Full pipeline as the default and list the Quick path as "minimal prediction only." | Either unblock Phase 5/6 from the simple path or set clear expectations. |
| 1.3 | **Have `models.py` write `phase3_top_models.json`** after Phase 3 grid search. Structure: `{"xgboost": [{"family": "xgboost", "params": "<json of best_params>", "cv_log_loss": ...}]}` so `phase4.py` can load best XGB params instead of using a fallback. | Aligns README flow with Phase 4’s expectations. |

---

## 2. Phase 1 — Data & ETL

| # | Fix | Why |
|---|-----|-----|
| 2.1 | **Add retries with backoff in `data-pull.py`** (e.g. 3 attempts with 2x backoff, then fail). Wrap `fetch_csv` / `fetch_html` in a small retry helper. | Reduces transient scrape failures. |
| 2.2 | **Validate Torvik schema after load.** After reading each Torvik CSV, assert expected column indices or names (e.g. that column 1 is team, 4 is adjoe, 44 is adjt), or read a small header row if the source provides one. Fail fast with a clear message if layout changed. | Catches Barttorvik layout changes before bad ETL. |
| 2.3 | **Optional: checksum or timestamp raw data.** Write `data/pull_manifest.json` with file paths and SHA-256 hashes (or last-modified) of each saved CSV after a successful pull. Document in README how to check reproducibility. | Makes "same data" verifiable. |
| 2.4 | **In `etl.py`, log join drop counts.** Before/after each merge (games→Torvik favorite, games→Torvik underdog, inference NET→Torvik), log how many rows were lost. | Surfaces name-matching and data coverage issues. |
| 2.5 | **Add symmetry assertion (or test).** In `etl.py` or `test_merge.py`: for a sample of rows, assert that the inverse row exists with negated diffs, inverted ratios, and flipped label. | Protects symmetric feature design from regressions. |

---

## 3. Phase 2 & 3 — Models (`models.py`)

| # | Fix | Why |
|---|-----|-----|
| 3.1 | **Use time-based CV for Phase 3 grid search.** Replace `cv=3` with a custom splitter: e.g. 3 folds where fold k uses years 2011..(2011+k) for train and next year for validation (or use `sklearn.model_selection.TimeSeriesSplit`-style folds on year). Pass it as `cv=<iterator>` to `GridSearchCV`. | Aligns hyperparameter selection with time-aware evaluation. |
| 3.2 | **Persist Phase 3 best params (see 1.3).** After `GridSearchCV`, write `data/phase3_top_models.json` with the best XGB params and holdout log loss so `phase4.py` does not rely on fallback. | Fixes README Phase 3 → Phase 4 handoff. |
| 3.3 | **Optional: multi-year holdout for reporting.** Add an option or separate script to evaluate Phase 2 and Phase 3 on multiple test years (e.g. 2014, 2015, 2016) with train = all years before test year; report mean and std of log loss / Brier. | Reduces variance and gives a more realistic performance picture. |

---

## 4. Phase 4 — Calibration

| # | Fix | Why |
|---|-----|-----|
| 4.1 | **Unify or clearly route Phase 4.** Choose one of: (a) Deprecate `phase4.py` and point README to `phase4_calibration.py` (and require `phase3_model_search.py`), or (b) Extend `phase4.py` to also write `phase4_oof_probs.csv`, `phase4_inference_probs.csv`, and `phase4_best_combos.json` (e.g. single-model "LR_isotonic" combo) so Phase 5 can run after the Quick pipeline. | Removes pipeline fragmentation. |
| 4.2 | **If keeping both:** Add a one-line comment at the top of `phase4.py` and `phase4_calibration.py` stating which pipeline (Quick vs Full) each belongs to and what the other script does. | Reduces confusion for future maintainers. |

---

## 5. Phase 5 — Ensemble

| # | Fix | Why |
|---|-----|-----|
| 5.1 | **Report ensemble holdout performance.** In Phase 5, load the same holdout year(s) used in Phase 4 (or a dedicated eval set), compute ensemble predictions from OOF→stack and BMA, and report log loss / Brier for the final blend on that holdout. Optionally log in `phase5_ensemble_weights.json`. | Validates the combined ensemble, not just in-sample meta loss. |
| 5.2 | **Optional: tune final blend weights.** Replace fixed (0.4, 0.4, 0.2) with a small search or closed-form fit on a validation slice (e.g. last year of OOF) to minimize log loss. | May improve calibration and log loss. |

---

## 6. Phase 6 — Simulation

| # | Fix | Why |
|---|-----|-----|
| 6.1 | **Document seed/team alignment.** In README or Phase 6 docstring, state that ensemble probs use `team_a`/`team_b` names and Phase 6 seed map (Kaggle or NET) must use the same naming for correct lookup; mention `normalize_name` or a shared team-ID layer if you add one. | Avoids silent 0.5 fallbacks for mismatched names. |
| 6.2 | **Optional: validate coverage.** After building `win_prob` lookup, log how many (team_a, team_b) pairs in the bracket have a non–default (0.5) probability; warn if below a threshold (e.g. 90%). | Surfaces name/ID mismatches between inference and bracket. |

---

## 7. Cross-cutting

| # | Fix | Why |
|---|-----|-----|
| 7.1 | **Single source of truth for "test year".** Define e.g. `TEST_YEAR = 2014` (or a list of holdout years) in a small `config.py` or at the top of one module, and have `models.py`, `phase4.py`, and `phase4_calibration.py` import it. | Ensures all phases use the same evaluation protocol. |
| 7.2 | **Optional: add a `run_all.py` or `Makefile`** that runs the Full pipeline in order (data-pull → etl → phase3_model_search → phase4_calibration → phase5 → phase6) with clear error messages if a prior artifact is missing. | One-command reproducibility for the full pipeline. |

---

## Priority summary

| Priority | Fixes | Effort (rough) |
|----------|-------|-----------------|
| **P0** | 1.1, 1.3, 3.2 (document pipelines; write phase3_top_models; persist XGB params) | Low |
| **P1** | 1.2 or 4.1 (either make Quick path feed Phase 5 or document Full as default) | Medium |
| **P1** | 2.4, 2.5 (ETL logging + symmetry assertion) | Low |
| **P2** | 2.1, 2.2 (retries, schema validation) | Low–medium |
| **P2** | 3.1 (time-based CV in Phase 3) | Medium |
| **P2** | 5.1 (ensemble holdout evaluation) | Low–medium |
| **P3** | 2.3, 3.3, 5.2, 6.1, 6.2, 7.1, 7.2 (reproducibility, multi-year eval, blend tune, docs, config, run_all) | Variable |

Implementing P0 and P1 fixes will resolve the main pipeline and handoff issues; P2 will strengthen methodology and robustness; P3 will improve long-term maintainability and reproducibility.
