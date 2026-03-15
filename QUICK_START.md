# March Madness ML - Quick Start Guide

## 🚀 Three Commands to Get Started

```bash
# Build datasets from raw Kaggle data
python build_master_dataset.py

# Create 2026 tournament prediction features  
python build_prediction_features.py

# Train baseline models
python baseline_model.py
```

---

## 📊 What Gets Created

### After `build_master_dataset.py`:

```
processed_data/
├── team_season_stats.csv          # Season-level stats for every team
├── tournament_games_features.csv  # Historical tournament games with features
└── dataset_summary.txt            # Quick stats overview
```

**Key Features Created:**
- Basic: Win%, PPG, PointDiff, Home/Away splits
- Advanced (2003+): Four Factors (eFG%, TOV%, ORB%, FTA rate)
- Tournament: Seeds, regions
- External: Massey ratings (POM, SAG, RPI, etc.)

### After `build_prediction_features.py`:

```
predictions/
├── prediction_features_2026.csv   # All possible 2026 matchup features
├── submission_template_2026.csv   # Kaggle submission format
└── feature_list.txt               # Documentation of all features
```

**Matchup Features:**
- All features are differentials (Team1 - Team2)
- ~50 features per matchup
- ID format: "2026_1104_1181" (Season_Team1_Team2)

### After `baseline_model.py`:

```
models/
├── baseline_results.csv                    # Model performance comparison
├── seed_matchup_rates.csv                  # Historical seed vs seed rates
└── logistic_regression_baseline.pkl        # Trained model (ready to use)
```

---

## 🎯 Quick Usage Examples

### Load and Explore Team Stats

```python
import pandas as pd

# Load team-season statistics
stats = pd.read_csv('processed_data/team_season_stats.csv')

# Get 2025 tournament teams
tourney_teams = stats[
    (stats['Season'] == 2025) & 
    (stats['SeedNum'].notna())
].sort_values('SeedNum')

print(tourney_teams[['TeamID', 'SeedNum', 'WinPct', 'eFG', 'PointDiff']].head(10))
```

### Make Predictions for Specific Matchup

```python
import pandas as pd
import joblib

# Load model and features
model = joblib.load('models/logistic_regression_baseline.pkl')
features = pd.read_csv('predictions/prediction_features_2026.csv')

# Get specific matchup (e.g., TeamID 1104 vs 1181)
matchup = features[
    (features['Team1ID'] == 1104) & 
    (features['Team2ID'] == 1181)
]

# Predict
feature_cols = ['Seed_Diff', 'WinPct_Diff', 'PointDiff_Diff', 'eFG_Diff', 'TOV_Diff']
prob = model.predict_proba(matchup[feature_cols].fillna(0))[:, 1][0]

print(f"Team 1104 win probability: {prob:.1%}")
print(f"Team 1181 win probability: {1-prob:.1%}")
```

### Create Full Kaggle Submission

```python
import pandas as pd
import joblib

# Load
model = joblib.load('models/logistic_regression_baseline.pkl')
X = pd.read_csv('predictions/prediction_features_2026.csv')

# Predict
feature_cols = ['Seed_Diff', 'WinPct_Diff', 'PointDiff_Diff', 'eFG_Diff', 'TOV_Diff']
predictions = model.predict_proba(X[feature_cols].fillna(0))[:, 1]

# Create submission
submission = pd.DataFrame({
    'ID': X['ID'],
    'Pred': predictions.clip(0.01, 0.99)  # Clip to avoid extreme log loss
})

submission.to_csv('my_submission.csv', index=False)
print(f"Created submission with {len(submission):,} predictions")
```

---

## 🔍 Explore Your Data

```bash
python explore_data.py
```

This script provides:
- Dataset summaries and shapes
- Sample data views
- Missing value analysis
- Historical upset analysis
- Seed performance statistics
- Data quality checks

---

## 📁 File Organization

**Input Data** (from Kaggle):
```
data/march-machine-learning-mania-2026/
├── MTeams.csv                           # Team reference
├── MNCAATourneyCompactResults.csv       # Historical tournament results
├── MNCAATourneySeeds.csv                # Tournament seeds
├── MRegularSeasonCompactResults.csv     # Regular season games
├── MRegularSeasonDetailedResults.csv    # Detailed box scores (2003+)
├── MTeamConferences.csv                 # Conference affiliations
└── MMasseyOrdinals.csv                  # External ratings (optional)
```

**Output Data** (created by scripts):
```
processed_data/    # Consolidated ML-ready datasets
predictions/       # 2026 prediction features and submission template
models/           # Trained models and results
```

---

## 🎓 Understanding the Features

### Basic Features (All Seasons: 1985+)
- `WinPct` - Regular season win percentage
- `PPG` - Points per game (offense)
- `OppPPG` - Opponent points per game (defense)  
- `PointDiff` - Average point differential
- `HomeWinPct`, `AwayWinPct` - Location splits

### Advanced Features (2003+)
**Four Factors** (Dean Oliver's framework):
- `eFG` - Effective FG% = (FGM + 0.5*3PM) / FGA
- `TOV_Rate` - Turnover % of possessions
- `ORB_Rate` - Offensive rebound % 
- `FTA_Rate` - Free throw rate (FTA/FGA)

### Tournament Features
- `SeedNum` - Tournament seed (1-16, null if didn't make it)
- `Region` - Tournament region (W, X, Y, Z)
- `ConfAbbrev` - Conference (acc, big_ten, sec, etc.)

### External Ratings (if available)
- `POM_Rank` - Pomeroy rating
- `SAG_Rank` - Sagarin rating
- `RPI_Rank` - Rating Percentage Index
- Lower rank = better team

---

## 🏆 Expected Performance

**Baseline Models (test set: 2022-2025):**

| Model | Log Loss | Accuracy | Description |
|-------|----------|----------|-------------|
| Random | 0.693 | 50% | Coin flip |
| Seed-Only | 0.52 | 67% | Historical seed matchup rates |
| Logistic Regression | 0.50 | 70% | Basic stats + seeds |
| **Target** | **<0.48** | **>72%** | Competitive submission |
| Top Kaggle | 0.42-0.46 | 75%+ | Advanced ensembles |

**Log loss is the scoring metric** - lower is better!

---

## ⚡ Pro Tips

1. **Always clip predictions to [0.01, 0.99]**
   - Extreme probabilities (0.0 or 1.0) cause infinite log loss
   - Even 99% confident = 0.99, not 1.0

2. **Use time-based validation**
   - Train on old seasons, test on recent seasons
   - Never use future data (data leakage)

3. **Tournament seeds are powerful**
   - Seed-only baseline often beats complex models
   - Combine seeds with other features for best results

4. **Don't ignore missing data**
   - Pre-2003 seasons lack detailed stats
   - Not all teams make tournament (SeedNum is null)
   - External ratings may be incomplete

5. **Ensemble for better results**
   - Combine seed model + logistic + XGBoost
   - Weighted average predictions
   - Reduces variance, improves stability

---

## 🐛 Common Issues

**"FileNotFoundError"**
- Make sure you're in project root directory
- Check that Kaggle data is in `data/march-machine-learning-mania-2026/`

**"KeyError" on features**
- Some features only exist for 2003+ (detailed stats)
- Use `.fillna(0)` or filter to recent seasons

**Predictions > 1.0 or < 0.0**
- Always clip: `predictions.clip(0.01, 0.99)`

**Model performs worse than baseline**
- Check for data leakage (using future data)
- Verify time-based train/test split
- Try simpler model or fewer features

---

## 📚 Next Steps

### Immediate Improvements:
1. **Add XGBoost model** - Usually outperforms logistic regression
2. **Feature engineering** - Create momentum, strength of schedule
3. **Hyperparameter tuning** - Use GridSearchCV or Optuna
4. **Model calibration** - Platt scaling for better probabilities

### Advanced Techniques:
1. **Team embeddings** - Neural network learned representations
2. **Bayesian models** - Uncertainty quantification
3. **External data** - KenPom, betting lines, player stats
4. **Ensemble stacking** - Meta-model combining multiple models

### Resources:
- Full documentation: `README_DATA_PIPELINE.md`
- Data exploration: `python explore_data.py`
- Kaggle forums: Competition discussion page

---

## ✅ Checklist

- [ ] Downloaded Kaggle data to `data/march-machine-learning-mania-2026/`
- [ ] Ran `python build_master_dataset.py`
- [ ] Ran `python build_prediction_features.py`
- [ ] Ran `python baseline_model.py`
- [ ] Reviewed baseline performance (models/baseline_results.csv)
- [ ] Explored data with `python explore_data.py`
- [ ] Created first submission
- [ ] Read full README for advanced tips
- [ ] Ready to improve models and climb leaderboard!

---

**Good luck with your March Madness predictions! 🏀**
