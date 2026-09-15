# 🏏 IPL Win Probability Predictor

Predicts the probability that the **chasing team** wins an IPL match from the current match situation at any point during the second innings.

The system is trained on **72,142 ball-level match situations from 623 IPL matches (2008–2019)** and evaluated chronologically on **116 completely unseen matches from IPL 2018–2019**.

The final deployed model is **Logistic Regression**, selected as an interpretable and probability-oriented model after comparing six machine-learning approaches.

---

## Problem Statement

During a T20 chase, the outcome remains uncertain until late in the innings.

Given the target, current score, balls and wickets remaining, required scoring rate, and recent momentum, the system estimates:

> **P(chasing team wins | current match situation)**

This is a binary classification problem where the **probability is the main product**, rather than simply predicting a hard 0/1 outcome.

For example:

```text
Chasing Team Win Probability: 68%
Bowling Team Win Probability: 32%
```

The application therefore exposes the model's probability directly through `predict_proba()`.

---

## Dataset

| File | Rows | Contents |
|---|---:|---|
| `matches (1).csv` | 756 | Match-level information including teams, city, venue, toss and winner |
| `deliveries 2.csv` | 179,078 | Ball-level runs, extras and dismissals |

The two datasets are joined using:

```text
matches.id == deliveries.match_id
```

The **first innings** provides the target score.

The **second innings** provides the match situations.

`matches.winner` provides the final binary target:

```text
1 = chasing team wins
0 = chasing team loses
```

### Cleaning Decisions

| Action | Count | Reason |
|---|---:|---|
| Dropped abandoned matches | 4 | No winner and therefore no valid target |
| Dropped D/L matches | 19 | Revised targets break the standard first-innings target assumption |
| Filled missing `city` | 7 | Recovered from venue information |
| Dropped `umpire3` | 84% missing | Rarely recorded and irrelevant to the chase |
| Dropped duplicate deliveries | 23 | Exact duplicate rows |
| Dropped short-lived franchises | 110 matches | Insufficient data for stable team effects |

### Team Normalization

Genuine franchise rebrandings were normalized:

```text
Delhi Daredevils → Delhi Capitals
Deccan Chargers → Sunrisers Hyderabad
Rising Pune Supergiants → Rising Pune Supergiant
```

After cleaning:

```text
623 matches
72,142 second-innings match situations
```

---

# Correct Cricket Calculations

Two important bugs from the original version were identified and corrected.

## 1. Legal Ball Counting

The `ball` column does not represent legal deliveries because wides and no-balls can create delivery numbers such as 7, 8 and 9 within an over.

The previous calculation:

```python
balls_left = 126 - (over * 6 + ball)
```

was incorrect.

The corrected implementation explicitly counts legal deliveries:

```python
is_legal_ball = (wide_runs == 0) & (noball_runs == 0)

legal_balls_bowled = (
    is_legal_ball
    .groupby(match_id)
    .cumsum()
)

balls_left = 120 - legal_balls_bowled
```

This correction is important because `balls_left` affects:

- Current Run Rate
- Required Run Rate
- Match pressure
- Win probability

The original calculation disagreed with the corrected legal-ball count on approximately **14.8% of second-innings rows**.

---

## 2. Cricket Over Notation

In cricket:

```text
17.5 overs = 17 completed overs + 5 legal balls
          = 107 legal balls
```

It does **not** mean:

```text
17.5 × 6 = 105 balls
```

The Streamlit application therefore accepts:

```text
Completed overs
Balls in current over
```

as separate integer inputs.

This prevents ambiguity in converting cricket notation into legal balls.

---

# Feature Engineering

The final model uses **19 features**.

### Match State

```text
batting_team
bowling_team
city
target_score
current_score
runs_left
balls_left
wickets
current_run_rate
required_run_rate
```

### Momentum

```text
runs_last_over
runs_last_3_overs
runs_last_5_overs
wickets_last_over
wickets_last_3_overs
```

### Pressure and Interaction Features

```text
score_progress_pct
runs_left_per_wicket
rrr_minus_crr
pressure_index
```

---

## Wickets

The model uses:

```text
wickets = wickets in hand
```

rather than wickets lost.

For example:

```text
0 wickets lost → 10 wickets in hand
3 wickets lost → 7 wickets in hand
6 wickets lost → 4 wickets in hand
```

This representation makes the feature directly useful for evaluating the resources available to the chasing team.

---

## Momentum Features Are Shift-Safe

Momentum is calculated only from **completed overs**.

For example, while predicting during the current over, the current over's unfinished runs are not included in:

```text
runs_last_over
runs_last_3_overs
runs_last_5_overs
```

This prevents information from the current prediction point from leaking into the features.

---

# Feature Reduction

Several candidate features were removed because they were essentially rescaled versions of existing features.

| Removed Feature | Existing Equivalent |
|---|---|
| `runs_left_per_ball` | `required_run_rate / 6` |
| `balls_progress_pct` | Rescaled ball progress |
| `wickets_remaining_ratio` | `wickets / 10` |
| `overs_completed` | Almost perfectly correlated with ball progress |

Retraining with and without these features showed that removing redundant features did not hurt performance.

Feature count itself was therefore not treated as a measure of model quality.

---

# Leakage Prevention

The target uses the final match winner.

The model features must not use any information that would only become known after the prediction point.

The following information was excluded:

```text
winner
result
win_by_runs
win_by_wickets
player_of_match
dismissal_kind
fielder
player_dismissed
batsman
bowler
non_striker
```

Final scores and future-delivery information were also excluded.

The notebook contains assertions to verify that forbidden columns do not enter the final feature set.

---

# Why Random Row Splitting Is Wrong

A match contains approximately 100+ delivery-level observations.

Consecutive observations from the same match are highly similar.

Therefore, a random row-level train/test split could put:

```text
Match A → training data
Match A → test data
```

at the same time.

That would allow the model to see extremely similar situations from the same match during training and testing.

The resulting score would therefore be overly optimistic.

---

# Chronological and Match-Safe Evaluation

The project instead uses a chronological split:

| Split | Seasons | Matches | Rows |
|---|---|---:|---:|
| Train | IPL 2008 → IPL 2016 | 477 | 55,194 |
| Validation | IPL 2017 | 30 | 3,309 |
| Test | IPL 2018 → IPL 2019 | 116 | 13,639 |

No match appears in more than one split.

The test seasons represent future data relative to the training period.

Hyperparameter evaluation inside the training data uses match-aware validation so that observations from the same match do not cross folds.

---

# Preprocessing

All preprocessing is implemented inside a scikit-learn `ColumnTransformer` and stored inside the final pipeline.

### Categorical Features

```text
OneHotEncoder(
    drop='first',
    handle_unknown='ignore'
)
```

One-hot encoding is used because teams and cities are nominal categories.

### Numerical Features

Extreme rate features are clipped:

```text
required_run_rate → maximum 36
pressure_index → maximum 18
```

These values are capped rather than deleting rows.

Extremely high required run rates are genuine cricket situations and contain useful information about difficult chases.

### Scaling

Numerical features are standardized for models that benefit from scaling:

```text
Logistic Regression → scaled
SVC → scaled
```

Tree-based models do not require scaling.

---

# Class Balance

The target is approximately balanced:

```text
Chasing team wins ≈ 52.7%
Chasing team loses ≈ 47.3%
```

Therefore:

- No SMOTE
- No artificial oversampling
- No class weighting

was required.

---

# Models Compared

Six machine-learning models were evaluated on the chronological test set:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Log Loss ↓ | Brier ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Random Forest | 0.7100 | 0.7956 | 0.6582 | 0.7205 | **0.8171** | **0.5472** | **0.1861** |
| XGBoost | **0.7161** | 0.7925 | 0.6772 | 0.7303 | 0.8138 | 0.5518 | 0.1876 |
| **Logistic Regression** | 0.7129 | 0.7897 | 0.6736 | 0.7271 | 0.8158 | 0.5621 | 0.1897 |
| Gradient Boosting | 0.6961 | 0.7681 | 0.6656 | 0.7132 | 0.7961 | 0.5980 | 0.2028 |
| SVC (RBF) | 0.6955 | 0.7576 | 0.6817 | 0.7176 | 0.7819 | 0.6760 | 0.2147 |
| Gaussian Naive Bayes | 0.6896 | 0.7059 | **0.7767** | 0.7396 | 0.7201 | 2.3849 | 0.2813 |

Baseline:

```text
Accuracy  ≈ 0.568
Log Loss  ≈ 0.688
Brier     ≈ 0.248
```

The baseline demonstrates that the model learns meaningful information beyond simply predicting the majority class.

---

# Final Model — Logistic Regression

The final deployed model is:

> **Logistic Regression**

The model was chosen for the final application because it provides:

- strong ROC-AUC
- interpretable coefficients
- stable probability-based predictions
- simple and efficient inference
- transparent feature effects
- straightforward deployment

Random Forest achieved slightly better probability metrics on the test benchmark, but Logistic Regression remained competitive:

```text
Logistic Regression ROC-AUC = 0.8158
Random Forest ROC-AUC       = 0.8171
```

The difference in ROC-AUC is small.

The final system therefore prioritizes **interpretability and simplicity** while retaining competitive predictive performance.

---

# Final Test Performance

The final deployed Logistic Regression model is evaluated on the completely unseen IPL 2018–2019 test seasons.

### Threshold = 0.50

| Metric | Score |
|---|---:|
| Accuracy | **71.29%** |
| Precision | **78.97%** |
| Recall | **67.36%** |
| F1 Score | **72.71%** |
| ROC-AUC | **81.58%** |
| Log Loss | **0.5621** |
| Brier Score | **0.1897** |

The model therefore provides useful ranking and probability information even though the classification accuracy is not artificially inflated through random row-level splitting.

---

# Threshold Selection

The classification threshold was selected using the **IPL 2017 validation season**.

The test set was not used to choose the threshold.

The evaluated thresholds included:

```text
0.20
0.25
0.30
0.35
0.40
0.45
0.50
0.55
0.60
0.65
0.70
0.75
0.80
```

The highest validation F1 occurred at:

```text
Threshold = 0.50
Validation F1 = 0.7463
```

Therefore:

```python
BEST_THRESHOLD = 0.50
```

was selected.

The test set was then evaluated once using this fixed threshold.

### Important distinction

The threshold does **not** change the probability displayed by the application.

The application uses:

```python
model.predict_proba()
```

to display the raw probability.

The threshold is only used when a hard classification is required:

```text
Probability ≥ 0.50 → Win
Probability < 0.50 → Loss
```

---

# Probability Output

The application displays both sides of the prediction.

Example:

```text
🏏 Mumbai Indians       — 63%
🏏 Chennai Super Kings  — 37%
```

The probabilities originate from the model:

```python
probabilities = model.predict_proba(match_state)[0]
```

The displayed percentages are rounded for readability while ensuring that the two displayed values sum to exactly 100%.

No manual probability smoothing or hardcoded prediction rules are used.

---

# Interpretability

Because the final model is Logistic Regression, feature importance is interpreted through the model's **coefficients** rather than tree impurity importance.

Positive coefficients increase the estimated probability of the chasing team winning.

Negative coefficients decrease it.

The notebook visualizes the largest absolute coefficients to show which encoded features have the strongest influence on the prediction.

SHAP is also implemented using a **linear explainer** appropriate for Logistic Regression rather than `TreeExplainer`.

This avoids applying a tree-specific explanation method to a linear model.

---

# Error Analysis

Prediction accuracy increases as the chase progresses:

| Chase Phase | Accuracy |
|---|---:|
| Early — 90–120 balls left | 0.635 |
| Middle — 60–89 balls left | 0.691 |
| Middle-late — 30–59 balls left | 0.773 |
| Death — 0–29 balls left | **0.837** |

This behaviour is expected.

Early in a chase, many possible outcomes remain open. A model producing extreme certainty with 100+ balls remaining would be suspicious.

As fewer balls remain, the match state becomes increasingly informative and predictions become easier.

---

# Match Progression

The notebook contains a match-progression analysis that calculates the predicted win probability over the course of a chase.

The IPL 2018 CSK vs MI opener is used as an example because it demonstrates how probability changes during a highly dynamic chase.

The model initially assigns a very low probability to CSK, but the probability rises sharply during the late stages as CSK scores heavily.

This example demonstrates that the model reacts to changing:

- score
- required run rate
- wickets
- remaining balls
- recent scoring momentum

rather than producing a static prediction.

---

# Cricket Insights

| Finding | Value |
|---|---:|
| Chasing teams win | **54.25%** |
| RRR ≤ 6 → chase succeeds | **96.5%** |
| RRR > 15 → chase succeeds | **3.2%** |
| 0–2 wickets in hand → win rate | **2.6%** |
| 9–10 wickets in hand → win rate | **65.4%** |
| Target ≤ 140 chased | **80.6%** |
| Target > 200 chased | **17.2%** |
| Toss winner wins | **51.69%** |

These statistics are descriptive findings from the dataset and should not be interpreted as causal relationships.

---

# Streamlit Application

Run the application using:

```bash
streamlit run app.py
```

The application loads:

```text
models/best_cricket_model.pkl
models/model_metadata.json
```

The model pipeline contains both preprocessing and Logistic Regression, ensuring that the preprocessing used by the application matches the preprocessing used during training.

---

## Application Inputs

The user provides:

```text
Batting Team
Bowling Team
City
Target Score
Current Score
Wickets Lost
Completed Overs
Balls in Current Over
```

Optional recent-momentum values can also be supplied.

The following features are derived automatically:

```text
Runs Left
Balls Left
Wickets in Hand
Current Run Rate
Required Run Rate
Score Progress
Runs Left per Wicket
RRR - CRR
Pressure Index
```

The user therefore does not need to manually calculate redundant features.

---

# Input Validation

The application prevents invalid match states such as:

- Same batting and bowling team
- Current score ≥ target
- More than 120 legal balls
- More than 10 wickets lost
- Negative runs remaining
- Invalid balls within an over
- Invalid momentum values

Only teams and cities present in the training metadata are offered to the user.

---

# Shared Feature Logic

`cricket_utils.py` is shared between the notebook and Streamlit application.

It contains the common logic for:

- Team normalization
- Active teams
- Feature column ordering
- Extreme-rate clipping
- Over-to-ball conversion
- Ball-to-over display conversion
- Match-state construction
- Preprocessing

This provides a **single source of truth** for feature engineering.

The saved pipeline depends on this module, so:

```text
cricket_utils.py
```

must remain alongside:

```text
app.py
```

when running the application.

---

# Project Structure

```text
.
├── cric_ai_pred.ipynb
├── app.py
├── cricket_utils.py
├── requirements.txt
├── matches (1).csv
├── deliveries 2.csv
└── models/
    ├── best_cricket_model.pkl
    └── model_metadata.json
```

### Important Files

`cric_ai_pred.ipynb`

> Complete data cleaning, feature engineering, chronological evaluation, model comparison, threshold selection, interpretability and error analysis.

`app.py`

> Streamlit interface for live win-probability prediction.

`cricket_utils.py`

> Shared feature engineering and preprocessing logic.

`best_cricket_model.pkl`

> Saved Logistic Regression pipeline containing preprocessing and the trained model.

`model_metadata.json`

> Stores model information, feature order, threshold, metrics, teams, cities and dataset split information.

---

# How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the notebook:

```bash
jupyter notebook cric_ai_pred.ipynb
```

Running the notebook from top to bottom regenerates:

```text
models/best_cricket_model.pkl
models/model_metadata.json
```

Launch the application:

```bash
streamlit run app.py
```

---

# Verification

The project includes verification checks for:

- Notebook execution
- Saved-model reload
- Feature-order consistency
- Probability output
- Match-level train/validation/test separation
- Leakage columns
- Notebook/app preprocessing consistency
- Valid cricket over-to-ball conversion
- Correct legal-ball counting
- Absence of the old `126 - (...)` formula
- Absence of incorrect decimal-over multiplication

The saved model successfully reloads and produces valid probabilities.

---

# Limitations

### 1. Early-Chase Uncertainty

Accuracy is lower early in the innings because many outcomes remain possible.

### 2. No Player-Level Features

The model does not know which batter or bowler is currently involved.

A specialist finisher or elite death bowler can therefore create situations the model has difficulty recognizing.

### 3. Scoring-Era Drift

Average target scores increased between the training and test periods.

The model therefore faces distribution shift when predicting later IPL seasons.

The historical dataset ends at IPL 2019.

### 4. Small Validation Season

IPL 2017 contains only 30 matches.

Threshold-selection results from this validation season are therefore noisy.

### 5. Historical Dataset

The training data covers IPL 2008–2019 and does not represent modern IPL scoring patterns.

### 6. D/L Matches

19 D/L matches were excluded because their revised targets do not fit the standard first-innings-target formulation used by this project.

---

# Future Improvements

Potential extensions include:

- Batter-specific historical strike rates
- Bowler-specific death-over performance
- Current batter/bowler identity
- Venue-specific par scores
- Season-aware retraining
- More recent IPL data
- Ball-level momentum modelling
- Batter partnership features
- Player form features
- Dynamic calibration using larger validation windows

---

# Key Takeaway

This project is designed around a realistic machine-learning evaluation setup rather than maximizing a single accuracy number.

The important engineering decisions are:

```text
Correct legal-ball calculation
        ↓
Leakage prevention
        ↓
Match-safe chronological split
        ↓
Feature engineering from prediction-time information
        ↓
Multiple model comparison
        ↓
Validation-based threshold selection
        ↓
Unseen future-season test evaluation
        ↓
Saved preprocessing + model pipeline
        ↓
Streamlit deployment
```

The final system achieves approximately:

```text
71.29% Test Accuracy
81.58% Test ROC-AUC
0.5621 Test Log Loss
0.1897 Test Brier Score
```

while maintaining an interpretable Logistic Regression model and a reproducible notebook-to-application pipeline.
