# BAT Plus: Regression-Weighted Batting Quality Model

A data-driven framework for measuring composite batting quality using regression-derived, zone-based contact and swing metrics. BAT Plus combines fast-swing rate, squared-up contact quality, and swing-decision index into a single, season-adjusted score.

## Model Architecture

### Three Components

**1. Fast Swing Rate (z-scored)**
- Raw measure of speed
- Standardized within each season to remove league-level shifts

**2. Contact Index (Regression-Derived)**
- **Why squared-up contact?** Squared-up contact directly measures contact *quality*
- **How it works**: Constrained OLS regression predicts squared-up contact rate from heart-zone, shadow-zone, and chase-zone contact rates.
- **The weights**: Fitted coefficients (Heart: +1.20, Shadow: +0.91, Chase: +0.30) show how each zone predicts actual squared-up contact.
- **Interpretation**: A high contact index means the player gets good contact quality.

**3. Decision Index (Regression-Derived)**
- **How it works**: Constrained OLS regression predicts walk percentage (BB%) from swing rates in heart, shadow, chase, and waste zones.
- **Target: Walk %**: Focuses on plate discipline
- **The weights** show marginal contribution to walk rate. Positive heart-zone, negative waste/chase reflects zone hierarchy in drawing walks.
- **Interpretation**: A high decision index means the player swings at hitter's pitches and lays off pitcher's pitches.

### Final Score

BAT Plus is a composite index scaled to league average = 100 within each season.

**Calculation**:
1. Fit linear regression: `pred = coef_swing * fast_swing_rate_z + coef_contact * contact_index_z + coef_decision * decision_index_z`
2. Compute season's mean prediction: `league_mean = mean(pred) by season`
3. Scale to 100-point index: `bat_plus = 100 * pred / league_mean`

**Interpretation**:
- **BAT Plus = 100**: Exactly league average for that season
- **BAT Plus = 110**: 10% above average
- **BAT Plus = 90**: 10% below average

```python
# In code:
data['pred_lin'] = LinearRegression().fit(X, y).predict(X)
league_mean = data.groupby('season')['pred_lin'].transform('mean')
data['bat_plus'] = 100 * data['pred_lin'] / league_mean
```

## Model Performance

- **In-sample r**: 0.4344 (predicting wOBA)
- **In-sample R²**: 0.1892
- **Cross-validation r (5-fold)**: 0.4222
- **Cross-validation R² (5-fold)**: 0.1798 ± 0.0370
- **Year-to-year autocorrelation (r) **: ~0.822 
- **Year-to-year autocorrelation (R²) **: ~0.676 (indicating strong skill stability)

| Bat Plus Range | Count | Average wOBA |
| :--- | :--- | :--- |
| 80-84 | 4 | 0.233 |
| 85-89 | 23 | 0.266 |
| 90-94 | 176 | 0.288 |
| 95-99 | 518 | 0.297 |
| 100-104 | 407 | 0.313 |
| 105-109 | 164 | 0.331 |
| 110-114 | 80 | 0.336 |
| 115-119 | 15 | 0.381 |
| 120-124 | 1 | 0.388 |

## Key Files

- **`batplus_notebook.ipynb`**: Complete end-to-end analysis
  - Data preparation and within-season standardization
  - Squared-up contact regression with constrained OLS
  - Swing-decision index derivation
  - BAT Plus model fit and diagnostics
  - Cross-validation summary
  - Year-to-year correlation analysis

- **`batplus_player_results.csv`**: Player-season export
  - Columns: `season`, `name`, `fast_swing_rate_z`, `contact_index_z`, `decision_index_z`, `bat_plus`, `obp`, `woba`
  - 1,764 complete player-season records

## Usage

### Requirements
```
pandas, numpy, scipy, sklearn, statsmodels, matplotlib
```
