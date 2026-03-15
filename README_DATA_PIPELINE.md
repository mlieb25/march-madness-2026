# March Madness Machine Learning Data Pipeline

## Overview

This repository contains a complete data processing and modeling pipeline for NCAA March Madness tournament predictions. The pipeline consolidates 40+ years of tournament data, regular season statistics, external ratings, and conference information into ML-ready datasets.

## Quick Start

```bash
# 1. Build master datasets
python build_master_dataset.py

# 2. Create prediction features for 2026
python build_prediction_features.py

# 3. Train baseline models
python baseline_model.py
```

## Pipeline Scripts

### 1. `build_master_dataset.py`

**Purpose:** Combines all raw data files into consolidated, feature-rich datasets.

**Inputs:**
- `MTeams.csv` - Team information
- `MNCAATourneyCompactResults.csv` - Tournament game results
- `MNCAATourneyDetailedResults.csv` - Detailed box scores (2003+)
- `MNCAATourneySeeds.csv` - Tournament seeding
- `MRegularSeasonCompactResults.csv` - Regular season results
- `MRegularSeasonDetailedResults.csv` - Detailed regular season stats
- `MTeamConferences.csv` - Conference affiliations
- `MMasseyOrdinals.csv` - External rating systems (optional)

**Outputs:**
- `processed_data/team_season_stats.csv` - Season-level team statistics
- `processed_data/tournament_games_features.csv` - Game-level features for training
- `processed_data/dataset_summary.txt` - Dataset summary

**Features Created:**

*Basic Statistics:*
- Wins, Losses, Win Percentage
- Points per game (offensive)
- Opponent points per game (defensive)
- Point differential
- Home/Away/Neutral win percentages

*Advanced Statistics (2003+):*
- **Four Factors:**
  - Effective FG% (eFG)
  - Free throw rate (FTA_Rate)
  - Turnover rate (TOV_Rate)
  - Offensive rebounding % (ORB_Rate)
- Total rebounds, assists, steals, blocks, personal fouls

*Tournament Information:*
- Tournament seed (1-16)
- Region assignment
- Conference affiliation

*External Ratings (if available):*
- Pomeroy (POM)
- Sagarin (SAG)
- RPI
- Dokter (DOK)
- Colley (COL)

**Example Usage:**
```python
import pandas as pd

# Load team-season stats
team_stats = pd.read_csv('processed_data/team_season_stats.csv')

# Get 2025 Duke statistics
duke_2025 = team_stats[
    (team_stats['Season'] == 2025) & 
    (team_stats['TeamID'] == 1181)
]
print(duke_2025[['TeamID', 'WinPct', 'PPG', 'SeedNum', 'eFG']])
```

---

### 2. `build_prediction_features.py`

**Purpose:** Creates pairwise matchup features for all possible team combinations in target season.

**Inputs:**
- `processed_data/team_season_stats.csv` (from build_master_dataset.py)
- `MTeams.csv`

**Outputs:**
- `predictions/prediction_features_2026.csv` - Full feature matrix for predictions
- `predictions/submission_template_2026.csv` - Kaggle submission format
- `predictions/feature_list.txt` - Complete feature documentation

**Features Created:**

All features are **differentials** (Team1 - Team2):
- `Seed_Diff` - Seed differential (negative = Team1 better seeded)
- `WinPct_Diff` - Win percentage differential
- `PointDiff_Diff` - Point differential differential
- `eFG_Diff` - Effective FG% differential
- `TOV_Diff` - Turnover rate differential
- `[Rating]_Rank_Diff` - External rating differentials
- `SameConference` - Binary indicator for conference matchup

**ID Format:**
```
ID = "Season_Team1ID_Team2ID"
Example: "2026_1104_1181" (Alabama vs Duke in 2026)
```

**Note:** Team1ID is always < Team2ID (Kaggle submission requirement)

**Example Usage:**
```python
import pandas as pd

# Load prediction features
pred_features = pd.read_csv('predictions/prediction_features_2026.csv')

# Get features for specific matchup (Alabama vs Duke)
matchup = pred_features[
    (pred_features['Team1ID'] == 1104) & 
    (pred_features['Team2ID'] == 1181)
]
print(matchup[['ID', 'Seed_Diff', 'WinPct_Diff', 'eFG_Diff']])
```

---

### 3. `baseline_model.py`

**Purpose:** Train and evaluate baseline prediction models.

**Inputs:**
- `processed_data/tournament_games_features.csv`

**Outputs:**
- `models/baseline_results.csv` - Performance comparison
- `models/seed_matchup_rates.csv` - Historical seed vs seed win rates
- `models/logistic_regression_baseline.pkl` - Trained model

**Models Implemented:**

1. **Seed-Based Model**
   - Uses historical win rates for each seed matchup (e.g., 1 vs 16, 8 vs 9)
   - Fills missing matchups with seed differential heuristic
   - Simple but surprisingly effective baseline

2. **Logistic Regression**
   - Features: Seed_Diff, WinPct_Diff, PointDiff_Diff, eFG_Diff, TOV_Diff
   - L2 regularization (default)
   - Interpretable coefficients

**Validation Strategy:**
- Time-based split: Train on 1985-2021, test on 2022-2025
- Ensures no future data leakage
- Realistic tournament prediction scenario

**Performance Metrics:**
- **Log Loss** - Primary Kaggle metric (lower is better)
- **Accuracy** - % of games predicted correctly
- **AUC** - Area under ROC curve

**Example Usage:**
```python
import pandas as pd
import joblib

# Load trained model
model = joblib.load('models/logistic_regression_baseline.pkl')

# Load prediction features
X = pd.read_csv('predictions/prediction_features_2026.csv')

# Make predictions
feature_cols = ['Seed_Diff', 'WinPct_Diff', 'PointDiff_Diff', 'eFG_Diff', 'TOV_Diff']
predictions = model.predict_proba(X[feature_cols].fillna(0))[:, 1]

# Create submission
submission = X[['ID']].copy()
submission['Pred'] = predictions
submission.to_csv('my_submission.csv', index=False)
```

---

## Directory Structure

```
march-madness/
├── data/
│   └── march-machine-learning-mania-2026/
│       ├── MTeams.csv
│       ├── MNCAATourneyCompactResults.csv
│       ├── MNCAATourneyDetailedResults.csv
│       ├── MNCAATourneySeeds.csv
│       ├── MRegularSeasonCompactResults.csv
│       ├── MRegularSeasonDetailedResults.csv
│       ├── MTeamConferences.csv
│       ├── MMasseyOrdinals.csv
│       └── ... (other files)
│
├── processed_data/
│   ├── team_season_stats.csv
│   ├── tournament_games_features.csv
│   └── dataset_summary.txt
│
├── predictions/
│   ├── prediction_features_2026.csv
│   ├── submission_template_2026.csv
│   └── feature_list.txt
│
├── models/
│   ├── baseline_results.csv
│   ├── seed_matchup_rates.csv
│   └── logistic_regression_baseline.pkl
│
├── build_master_dataset.py
├── build_prediction_features.py
├── baseline_model.py
└── README_DATA_PIPELINE.md
```

---

## Data Schema Reference

### Team Season Stats Schema

| Column | Type | Description |
|--------|------|-------------|
| Season | int | Year of season |
| TeamID | int | Unique team identifier |
| Wins | int | Regular season wins |
| Games | int | Total regular season games |
| WinPct | float | Win percentage (0.0-1.0) |
| PPG | float | Points per game (offense) |
| OppPPG | float | Opponent points per game (defense) |
| PointDiff | float | Average point differential |
| HomeWinPct | float | Win % at home |
| AwayWinPct | float | Win % away |
| eFG | float | Effective field goal % |
| FTA_Rate | float | Free throw rate |
| TOV_Rate | float | Turnover rate |
| ORB_Rate | float | Offensive rebounding % |
| SeedNum | int | Tournament seed (1-16, null if didn't make tournament) |
| ConfAbbrev | str | Conference abbreviation |

### Tournament Games Features Schema

| Column | Type | Description |
|--------|------|-------------|
| Season | int | Year of tournament |
| DayNum | int | Day of season (136 = tournament start) |
| WTeamID | int | Winning team ID |
| WScore | int | Winning team score |
| LTeamID | int | Losing team ID |
| LScore | int | Losing team score |
| WLoc | str | Winner location (N=neutral for tourney) |
| NumOT | int | Number of overtime periods |
| SeedDiff | int | Winner seed - Loser seed |
| WinPctDiff | float | Winner WinPct - Loser WinPct |
| PointDiffDiff | float | Winner PointDiff - Loser PointDiff |
| eFGDiff | float | Winner eFG - Loser eFG |
| ... | ... | Many more differential features |

---

## Advanced Modeling Tips

### Feature Engineering Ideas

1. **Momentum Features**
   - Win streak length
   - Recent performance (last 10 games)
   - Conference tournament performance

2. **Strength of Schedule**
   - Average opponent rating
   - Quality wins (vs top 25)
   - Bad losses (vs below .500)

3. **Playing Style Matchups**
   - Tempo differential (possessions per game)
   - 3-point shooting vs 3-point defense
   - Offensive efficiency vs defensive efficiency

4. **Experience Features**
   - Previous tournament appearances
   - Returning players (would need external data)
   - Coaching experience

### Model Ensemble Strategy

```python
# Combine multiple models
from sklearn.ensemble import VotingClassifier

# Create ensemble
ensemble = VotingClassifier(
    estimators=[
        ('lr', logistic_model),
        ('rf', random_forest_model),
        ('xgb', xgboost_model)
    ],
    voting='soft',  # Use probability predictions
    weights=[1, 2, 3]  # Weight better models more
)
```

### Cross-Validation Best Practices

```python
# Time-series CV for temporal data
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)

for train_idx, test_idx in tscv.split(data):
    train = data.iloc[train_idx]
    test = data.iloc[test_idx]
    # Train and evaluate
```

---

## Kaggle Submission Format

```csv
ID,Pred
2026_1101_1102,0.45
2026_1101_1103,0.62
2026_1101_1104,0.38
...
```

**Requirements:**
- `ID`: Format `Season_TeamID1_TeamID2` where TeamID1 < TeamID2
- `Pred`: Probability Team1 beats Team2 (0.0 to 1.0)
- All possible pairwise matchups for active teams
- Clip predictions to [0.01, 0.99] to avoid extreme log loss penalties

---

## Troubleshooting

**Issue:** `FileNotFoundError` when running scripts
- **Solution:** Ensure you're in the project root directory and data files exist

**Issue:** Missing values in features
- **Solution:** Pre-2003 seasons don't have detailed stats. Use `.fillna(0)` or filter to 2003+

**Issue:** Teams without seeds
- **Solution:** These teams didn't make the tournament. Filter them out or use `.dropna(subset=['SeedNum'])`

**Issue:** Model overfitting
- **Solution:** Use time-based CV, reduce features, add regularization

---

## Performance Benchmarks

Typical baseline performance (test set: 2022-2025):

| Model | Log Loss | Accuracy | Notes |
|-------|----------|----------|-------|
| Random (0.5) | 0.693 | ~50% | Theoretical worst |
| Seed-Only | 0.50-0.55 | 65-70% | Strong baseline |
| Logistic Regression | 0.48-0.52 | 68-72% | Good interpretable model |
| XGBoost | 0.45-0.50 | 70-75% | Advanced tree model |
| Ensemble | 0.44-0.48 | 72-76% | Best performance |

**Kaggle Leaderboard Context:**
- Top submissions typically achieve 0.42-0.48 log loss
- Winning solutions often ensemble 5-10 models
- Domain expertise (basketball knowledge) helps significantly

---

## Next Steps

1. **Data Enhancement**
   - Add KenPom ratings (external source)
   - Incorporate betting lines (market probabilities)
   - Include coach tenure/experience

2. **Advanced Models**
   - XGBoost with hyperparameter tuning
   - Neural networks with team embeddings
   - Bayesian hierarchical models

3. **Feature Selection**
   - Correlation analysis
   - Recursive feature elimination
   - SHAP values for interpretability

4. **Probability Calibration**
   - Platt scaling
   - Isotonic regression
   - Beta calibration

---

## References

- **Kaggle Competition:** [March Machine Learning Mania](https://www.kaggle.com/c/march-machine-learning-mania-2026)
- **KenPom:** [kenpom.com](https://kenpom.com) - Advanced basketball analytics
- **Basketball Reference:** [sports-reference.com](https://www.sports-reference.com/cbb/)
- **Four Factors:** Dean Oliver's "Basketball on Paper"

---

## Contact

Mitchell Liebrecht  
march-madness project  
March 15, 2026
