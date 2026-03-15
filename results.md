# March Madness ML Project Results

## Phase 1: Feature Factory Updates (ETL)
The `etl.py` script was expanded to fully satisfy the "Feature Factory" requirements in the project procedure document. 

The custom script maps disparate names across historical datasets and enforces **Strict Symmetry**. When games are observed from team A's perspective versus team B's, their difference/ratio metrics perfectly mirror each other (negated or inversed) so that the ML algorithm doesn't over-index on raw team assignments.

### Training Data Generation
The pipeline generated an ML training dataset that utilizes FiveThirtyEight historical March Madness game outcomes as the ground truth `y` value (`favorite_win_flag` = 1 or 0). 

It generates 12 complex features based on Barttorvik's historical stat-lines:
- **Differentials** (`adjoe_diff`, `adjde_diff`, `barthag_diff`, `sos_diff`, `wab_diff`, `adjt_diff`)
- **Ratios** (`adjoe_ratio`, `adjde_ratio`, `barthag_ratio`, `sos_ratio`, `wab_ratio`, `adjt_ratio`)

**Output File**: `data/ml_training_data.csv`
- Total Row Count: 474 (completely balanced with swapped inverse matches) 

### Inference Data Generation (2026 Predictions)
To prepare for predictions directly on this year's brackets, the pipeline isolated the top 68 teams according to current NCAA NET rankings. It bound them securely to their current 2026 Torvik stats and mapped out every possible pairwise matchup out of those 68 teams.

**Output File**: `data/ml_inference_data_2026.csv`
- Total Matchups Generated: 1,953 possible games. (Successfully matched 63/68 teams)
- These future matchups use the exact same 12-feature schema as the training data.

---

## Phase 2 & 3: ML Modeling

The Baseline Logistic Modeling (Phase 2) and Systematic Tree Ensemble Search (Phase 3) were built into `models.py`.
- The pipeline utilizes `scikit-learn` and `xgboost`.
- The evaluation employs a strict time-aware split to prevent data leakage. The models were trained on games from 2011 to 2013 and predicted blindly on 2014.

### Phase 2: Baseline Logistic Regression
A standard Logistic Regression model scaled by `StandardScaler` was used to establish the baseline for the project.

**Holdout Performance:**
- **Log Loss**: `0.5040`
- **Brier Score**: `0.1658`

### Phase 3: Systematic Model Search (Multi-Family)
A comprehensive search across multiple model families (Elastic Net, XGBoost, LightGBM, Gaussian Processes) was executed using Optuna for hyperparameter optimization.

**Top CV Log Loss per Family:**
- **LightGBM**: `0.5146`
- **Elastic Net**: `0.5182`
- **XGBoost**: `0.5396`
- **GP**: `0.5303`

### Phase 4: Advanced Model Calibration
Using multiple advanced calibration techniques (Platt, Isotonic, Beta, Venn-Abers), the top models were adjusted for probabilistic sharpness.

**Selected Best Combos (last season holdout):**
- **LightGBM + Isotonic**: Log Loss **0.48389** | Brier Score **0.16021**
- **Elastic Net + Beta**: Log Loss **0.49712** | Brier Score **0.16531**
- **GP + Beta**: Log Loss **0.50377** | Brier Score **0.16856**


## Phase 5: Ensembling & BMA
The final consensus model was built by blending the top-performing calibrated models (**LightGBM**, **Elastic Net**, and **Gaussian Process**) using Bayesian Model Averaging (BMA) and stacking.

**Final Ensemble Performance (Historical OOF Growth):**
- **LightGBM (Isotonic)**: 47.11x Bankroll growth
- **Ensemble (BMA)**: 42.58x Bankroll growth

**Output File**: `data/phase5_ensemble_probs.csv`

## Phase 6: Tournament Simulation & Strategy
10,000 Monte Carlo simulations were run using the final ensemble probabilities to determine tournament reach and optimize bracket strategies.

**Top 5 Championship Contenders:**
1. **Michigan**: 15.37%
2. **Duke**: 15.35%
3. **Florida**: 11.67%
4. **Arizona**: 10.92%
5. **Houston**: 6.64%

**Bracket Strategies (Mean Pool EV):**
- **Chalk**: 189.53 pts
- **Exploitative**: 185.63 pts
- **High Variance**: 171.79 pts

**Key Output Files**: 
- `data/phase6_brackets.json` (Optimized picks)
- `data/phase6_simulation_plots.png` (Visualization)
