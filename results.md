# March Madness ML Project Results

*Last full pipeline run: Quick path (data-pull → etl → models → phase4 → phase5 → phase6).*

---

## Phase 1: Feature Factory (ETL)

The `etl.py` script maps disparate names across historical datasets and enforces **strict symmetry**: when games are observed from team A's perspective versus team B's, difference/ratio metrics mirror (negated or inverted) so the model does not over-index on team order.

### Training Data
- **Source:** FiveThirtyEight historical tournament outcomes (`favorite_win_flag`), Barttorvik efficiency stats.
- **Features:** 12 pairwise features — differentials and ratios for adjOE, adjDE, Barthag, SoS, WAB, Tempo.
- **Join counts (this run):**
  - Games → Torvik favorite: 253 → 251 rows (dropped 2)
  - Games → Torvik underdog: 251 → 237 rows (dropped 14)
- **Output:** `data/ml_training_data.csv` — **474 rows** (symmetric original + inverse). Symmetry assertion passed.

### Inference Data (2026)
- **Output:** `data/ml_inference_data_2026.csv`
- **Matchups:** 1,953 possible games
- **Teams matched:** 63 of 68 (NET → Torvik 2026); 5 dropped at join.

---

## Phase 2 & 3: ML Modeling

Evaluation uses a **time-aware split** (config: `TEST_YEAR = 2014`): train on years &lt; 2014, holdout 2014. Phase 3 uses time-based CV when possible; this run fell back to KFold (3 folds) for hyperparameter search.

### Phase 2: Baseline Logistic Regression
- **Holdout (2014):** Log Loss **0.5040** | Brier **0.1658** | ROC-AUC 0.8407
- **Outputs:** `data/baseline_predictions_2026.csv`, `data/models/baseline_lr.pkl`

### Phase 3: XGBoost Grid Search
- **Best params:** `learning_rate=0.01`, `max_depth=2`, `n_estimators=200`, `subsample=1.0`
- **Best CV log loss:** 0.5876
- **Holdout (2014):** Log Loss **0.5207** | Brier **0.1698** | ROC-AUC 0.8155
- **Outputs:** `data/xgb_predictions_2026.csv`, `data/models/xgb_best.pkl`, `data/phase3_top_models.json`

### Multi-year holdout (LR only, this run)
- 2014: log loss 0.5040 (single year available in split)

---

## Phase 4: Calibration (Quick pipeline)

Single-split calibration (train &lt; 2014, test 2014). Bridge files written for Phase 5.

### Logistic Regression
| Variant              | Log Loss | Brier  |
|----------------------|----------|--------|
| Uncalibrated         | 0.5040   | 0.1658 |
| Isotonic calibrated  | 0.5244   | 0.1684 |
| Sigmoid calibrated   | 0.5158   | 0.1698 |

### XGBoost
| Variant              | Log Loss | Brier  |
|----------------------|----------|--------|
| Uncalibrated         | 0.5207   | 0.1698 |
| Isotonic calibrated  | 0.5733   | 0.1860 |
| Sigmoid calibrated   | 0.6085   | 0.2097 |

Final 2026 predictions use **Isotonic-calibrated LR**. Outputs: `data/calibrated_predictions_2026.csv`, `data/phase4_oof_probs.csv`, `data/phase4_inference_probs.csv`, `data/phase4_best_combos.json`.

---

## Phase 5: Ensembling & BMA

Quick path: single base model **logistic_isotonic**. Blend weights tuned on holdout year.

- **Final blend:** BMA 0.40, Stack 0.46, Risk-adaptive 0.14
- **Ensemble holdout (2014, n=96):** Log Loss **0.5140** | Brier **0.1674**
- **Kelly bankroll (start=1.0):** logistic_isotonic 139.05×, BMA ensemble 139.05×
- **Outputs:** `data/phase5_ensemble_probs.csv`, `data/phase5_ensemble_weights.json`, `data/phase5_kelly_results.csv`, `data/phase5_submission.csv`

---

## Phase 6: Tournament Simulation & Strategy

- **Simulations:** 10,000
- **Seeding:** NET fallback (64 slots)
- **Bracket matchup coverage:** 1,711 / 2,016 (84.9%) with non-default prob (warning: below 90% — seed/team names may not fully match ensemble probs)

### Top 10 championship contenders
| Rank | Team       | Elite 8 | Final Four | Championship |
|------|------------|---------|------------|---------------|
| 1    | Duke       | 47.9%   | 52.3%      | **13.3%**     |
| 2    | Illinois   | 49.8%   | 50.9%      | **11.9%**     |
| 3    | Florida    | 39.4%   | 42.6%      | **10.6%**     |
| 4    | Michigan   | 45.7%   | 50.5%      | **10.4%**     |
| 5    | Houston    | 45.6%   | 45.8%      | **10.2%**     |
| 6    | Iowa St.   | 52.5%   | 50.9%      | 9.5%          |
| 7    | Arizona    | 36.3%   | 36.7%      | 6.5%          |
| 8    | Purdue     | 29.6%   | 25.2%      | 4.4%          |
| 9    | Texas Tech | 22.5%   | 19.0%      | 2.1%          |
| 10   | NC State   | 12.1%   | 9.6%       | 1.8%          |

### Pool EV (scoring 1,2,4,8,16,32; 5,000 sims)
| Bracket        | Mean EV | Std  | p10  | p50  | p90  | p99  |
|----------------|---------|------|------|------|------|------|
| Exploitative   | 183.41  | 27.13| 151.8| 180.0| 222.0| 254.0|
| Chalk          | 182.61  | 27.55| 150.0| 180.0| 220.0| 256.0|
| High variance  | 170.32  | 24.25| 140.0| 168.0| 204.0| 234.0|

### Key upset paths (seed ≥ 10, P(Sweet 16) notable)
- South Fla. (13), McNeese (14), Cincinnati (12), Santa Clara (10), SMU (10), Indiana (11), Auburn (10), Texas A&M (11), Texas (11), Oklahoma (12)

**Outputs:** `data/phase6_team_round_probs.csv`, `data/phase6_brackets.json`, `data/phase6_pool_ev.csv`, `data/phase6_upset_paths.csv`, `data/phase6_simulation_plots.png`, `data/phase6_simulation_raw.csv`.
