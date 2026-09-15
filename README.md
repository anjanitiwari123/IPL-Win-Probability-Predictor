# 🏏 IPL Win Probability Prediction System

Predicts the probability that the **chasing team** wins an IPL match, from the match
situation at any point in the second innings.

Trained on 72,142 ball-level match situations from 623 IPL matches (2008–2019),
evaluated on 116 matches the model never saw.

---

## Problem statement

During a T20 chase, the outcome is genuinely uncertain until late. Given the target,
the current score, the balls and wickets remaining, and how the last few overs have
gone, we want:

> **P(chasing team wins | current match situation)**

This is a binary classification problem where **the probability is the product**, not
the hard 0/1 label. A model that says "68% / 32%" is useful; one that says "win" is
not. That framing drives every decision below — most importantly, the model is selected
on log loss and Brier score rather than accuracy.

---

## Dataset

| File | Rows | Contents |
|---|---|---|
| `matches (1).csv` | 756 | One row per match: teams, city, venue, toss, winner, D/L flag |
| `deliveries 2.csv` | 179,078 | One row per ball: runs, extras, dismissals |

The two join on `matches.id == deliveries.match_id`. The **first innings** supplies the
target, the **second innings** supplies the match situations, and `matches.winner`
supplies the label.

### Cleaning decisions

Every removal has a stated reason; nothing was dropped just to tidy the data.

| Action | Count | Reason |
|---|---|---|
| Dropped abandoned matches | 4 | `result = "no result"`, no winner → no target can be built |
| Dropped D/L matches | 19 | Revised targets break the "target = first-innings total" assumption |
| Filled missing `city` | 7 | All were at Dubai International Cricket Stadium — recovered from `venue`, not guessed |
| Dropped `umpire3` column | 84% missing | Third umpire rarely recorded, irrelevant to a chase |
| Dropped duplicate delivery rows | 23 | Exact duplicates |
| Dropped short-lived franchises | 110 matches | Gujarat Lions, Kochi Tuskers, Pune Warriors, Rising Pune — 1–3 seasons each, too little data for a stable team effect |

`player_dismissed`, `dismissal_kind` and `fielder` are ~95% missing. This is **not**
dirty data — a wicket does not fall on most balls. They were converted to a binary
wicket flag rather than imputed.

**Team normalization** (genuine rebrandings only): Delhi Daredevils → Delhi Capitals,
Deccan Chargers → Sunrisers Hyderabad, and a "Rising Pune Supergiants/Supergiant"
spelling fix.

**Final dataset: 623 matches, 72,142 second-innings rows.**

---

## Correct cricket calculations

Two bugs in the original version of this project produced wrong features. Both are
fixed and the fix is verified in the notebook.

### 1. Legal-ball counting

The `ball` column counts **every delivery bowled**, including wides and no-balls — it
reaches 7, 8 and 9. The original formula was:

```python
balls_left = 126 - (over * 6 + ball)      # WRONG
```

This over-counts balls whenever an over contains an extra. Measured against the correct
count, **it disagrees on 14.8% of all second-innings rows (10,706 of 72,378)**. Because
`balls_left` feeds both CRR and RRR, three features were corrupted together.

The fix counts legal balls explicitly:

```python
is_legal_ball = (wide_runs == 0) & (noball_runs == 0)
legal_balls_bowled = is_legal_ball.groupby(match_id).cumsum()
balls_left = 120 - legal_balls_bowled
```

### 2. Over notation

`17.5 overs` means 17 completed overs **plus 5 balls = 107 legal balls**. It does not
mean `17.5 × 6 = 105`. The old Streamlit app made exactly this mistake. The new app
takes completed overs and balls-in-current-over as two separate whole numbers, so the
ambiguity cannot arise.

---

## Feature engineering

19 features, all computable from information available at prediction time.

**Match state:** `batting_team`, `bowling_team`, `city`, `target_score`,
`current_score`, `runs_left`, `balls_left`, `wickets`, `current_run_rate`,
`required_run_rate`

**Momentum:** `runs_last_over`, `runs_last_3_overs`, `runs_last_5_overs`,
`wickets_last_over`, `wickets_last_3_overs`

**Pressure / interaction:** `score_progress_pct`, `runs_left_per_wicket`,
`rrr_minus_crr`, `pressure_index`

### Momentum features are shift-safe

"Runs in the last 3 overs" is only legitimate if it uses overs that were **already
finished**. Including the current over would leak runs being scored at the moment of
prediction. Each window is therefore computed on a cumulative total shifted by one full
over, so only completed overs contribute.

### Features deliberately removed

Four candidates turned out to be **exact rescalings** of features already present
(correlation 1.000):

| Removed | Equals |
|---|---|
| `runs_left_per_ball` | `required_run_rate / 6` |
| `balls_progress_pct` | `(120 - balls_left) / 1.2` |
| `wickets_remaining_ratio` | `wickets / 10` |
| `overs_completed` | `balls_left` rescaled (r = 0.999) |

Retrained with and without them: dropping them made the model **slightly better** on
the test seasons. Feature count is not a quality metric.

### One feature not created

An `rrr / crr` ratio. On the first ball of an innings `current_run_rate` is exactly 0,
making the ratio undefined for a real and common situation. `rrr_minus_crr` carries the
same signal without the division by zero.

---

## Leakage prevention

The label may use `winner`. **The features may not.** Columns permanently excluded:
`winner`, `result`, `win_by_runs`, `win_by_wickets`, `player_of_match`,
`dismissal_kind`, `fielder`, `player_dismissed`, `batsman`, `bowler`, `non_striker`,
and any final-score or future-delivery information.

The notebook enforces this with an assertion rather than trusting the author's memory,
and additionally verifies that the label is constant within every match and that no
numeric feature correlates near-perfectly with the target.

### Why a random row split would be wrong

Each match contributes ~110 delivery rows, and consecutive rows are nearly identical.
A random row-level split places **the same match** in both train and test, so the model
recognises situations it has already seen and the score measures memorisation.

### The split actually used — chronological and match-safe

| Split | Seasons | Matches | Rows |
|---|---|---:|---:|
| Train | IPL-2008 → IPL-2016 | 477 | 55,194 |
| Validation | IPL-2017 | 30 | 3,309 |
| Test | IPL-2018 + IPL-2019 | 116 | 13,639 |

Stricter than a group split: no match spans two sets, **and** the model never learns
from the future to predict the past. Hyperparameter search inside training data uses
`GroupKFold` on `match_id`. The test set was touched once, at the very end.

---

## Preprocessing

All preprocessing lives inside a scikit-learn `ColumnTransformer`, fitted on training
data only and carried inside the saved artifact — so the app cannot preprocess
differently from the notebook.

- **Categoricals** → `OneHotEncoder(drop='first', handle_unknown='ignore')`. One-hot
  rather than `LabelEncoder` because teams and cities are nominal.
- **Extreme rates** → `required_run_rate` capped at 36 (a six off every remaining ball
  is the theoretical maximum), `pressure_index` capped at 18.
- **Scaling** → applied only where the model needs it:

| Model | Scaled | Why |
|---|---|---|
| Logistic Regression, SVC | yes | coefficient / distance based |
| Naive Bayes | no | fits a Gaussian per feature |
| Random Forest, Gradient Boosting, XGBoost | no | trees split on thresholds |

### Outliers were capped, not deleted

Required run rates run up to 678. These are **real** death-over situations, not corrupt
data — 99.8% of them end in a loss, exactly as cricket sense predicts. **Not one row was
removed.** Capping preserves both the row and its "hopeless chase" signal while
compressing the numeric tail for the scale-sensitive models.

### Class imbalance: none

Target distribution is **52.7% / 47.3%** — effectively balanced. No SMOTE, no class
weights. Chasing teams genuinely win a little over half of IPL matches.

---

## Models compared

All six were actually trained and evaluated. Results on the **116 unseen test matches**:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Log Loss ↓ | Brier ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Random Forest** | 0.7100 | 0.7956 | 0.6582 | 0.7205 | **0.8171** | **0.5472** | **0.1861** |
| XGBoost | 0.7161 | 0.7925 | 0.6772 | 0.7303 | 0.8138 | 0.5518 | 0.1876 |
| Logistic Regression | 0.7129 | 0.7897 | 0.6736 | 0.7271 | 0.8158 | 0.5621 | 0.1897 |
| Gradient Boosting | 0.6961 | 0.7681 | 0.6656 | 0.7132 | 0.7961 | 0.5980 | 0.2028 |
| SVC (RBF) | 0.6955 | 0.7576 | 0.6817 | 0.7176 | 0.7819 | 0.6760 | 0.2147 |
| Gaussian Naive Bayes | 0.6896 | 0.7059 | 0.7767 | 0.7396 | 0.7201 | 2.3849 | 0.2813 |

Baseline (always predict the base rate): accuracy 0.5676, log loss 0.6881, Brier 0.2475.

**Why Naive Bayes proves the point.** Its accuracy (0.690) looks respectable and its
recall is the highest in the table — but its log loss is **catastrophic (2.38)**. It
assumes features are independent given the class, which is badly false here, so it
emits wildly overconfident probabilities. Selecting on accuracy would have shipped a
model that is useless as a probability estimator.

*SVC note:* an RBF SVC is O(n²), so it was fitted on an 8,000-row sample of the training
set. That is a real limitation of the model at this data size, reported rather than
hidden.

### Hyperparameter tuning

Moderate `RandomizedSearchCV` (12 candidates) for XGBoost with `GroupKFold` on
`match_id`, scored on `neg_log_loss`; a small explicit candidate list for Random Forest
and Gradient Boosting scored on the validation season. The test set was never involved.

**Untuned XGBoost was initially the worst model in the table** (log loss 0.83). That was
a hyperparameter problem, not an algorithm problem — which is why every model was given
a fair shot before any conclusion was drawn.

---

## Probability calibration — evaluated, then rejected

Raw probabilities were compared against Platt scaling and isotonic regression, with the
calibrator fitted on the validation season and judged on test:

| Variant | Log Loss ↓ |
|---|---:|
| **Random Forest, raw** | **0.5472** |
| Random Forest, sigmoid | 0.6306 |
| Random Forest, isotonic | 1.1483 |

Both methods made things **worse**. Isotonic is flexible enough to overfit the small
30-match validation season and its test log loss blows up. The raw Random Forest curve
already tracks the diagonal closely on the reliability diagram.

**The final model ships with raw `predict_proba` and no calibration layer.** No
clipping, no shrinking toward 50%, no manual adjustment of any kind.

---

## Model selection

Random Forest ranks **first on ROC-AUC, log loss and Brier score simultaneously**.

Two choices made against the more impressive-sounding option:

- **Not XGBoost.** It edges ahead on raw accuracy (0.7161 vs 0.7100) but loses on all
  three probability metrics — and accuracy is not what this system sells.
- **Not calibrated.** The evidence said calibration hurt.

Logistic Regression finishes a close third and would be a defensible ship if
interpretability were the priority.

### Final metrics — Random Forest, threshold 0.40

| Metric | Value |
|---|---:|
| ROC-AUC | **0.8171** |
| Log Loss | **0.5472** |
| Brier Score | **0.1861** |
| Accuracy | 0.7294 |
| Precision | 0.7682 |
| Recall | 0.7493 |
| F1 | 0.7586 |

Threshold 0.40 was selected on the **validation** season (not on test), improving test
F1 from 0.721 to 0.759. The app still displays the raw probability; the threshold is
stored in metadata for anyone needing a hard win/loss call.

---

## Interpretability

SHAP `TreeExplainer` on the selected Random Forest. Top drivers by mean |SHAP|:

1. **`required_run_rate`** — the strongest single driver
2. **`rrr_minus_crr`** — ahead of or behind the rate the team has been scoring at
3. **`runs_left_per_wicket`** — what each remaining batter owes
4. **`pressure_index`** — required rate discounted by wickets in hand
5. **`target_score`** — the run-scoring environment of that match

Then, well behind: `wickets`, `runs_left`, and the team/city columns.

**The engineered interactions outrank the raw inputs they were built from.**
`runs_left` alone is only 7th, but `runs_left_per_wicket` is 3rd — the model cares less
about *how many runs* than about *how many runs relative to the resources left*, which
is how a commentator reads a chase. Team and city effects are small: the situation
dominates, not the badge.

*Permutation importance is also reported and comes out mostly near-zero or negative.
That is the standard correlated-feature artifact — shuffling `required_run_rate` leaves
`pressure_index` carrying the same information — not evidence the features are useless.
SHAP is the ranking to trust here.*

---

## Error analysis

**Accuracy rises as the chase progresses**, which is the correct shape:

| Phase | Accuracy |
|---|---:|
| Early (90–120 balls left) | 0.635 |
| Middle (60–89) | 0.691 |
| Middle-late (30–59) | 0.773 |
| Death (0–29) | **0.837** |

Early in a chase the outcome genuinely is near a coin flip; a model claiming confidence
there would be lying.

By wickets in hand: 0–2 → 0.909 (nearly decided), 9–10 → 0.672 (mostly early
situations).

**Confident predictions (>90% or <10%) are wrong 7.1% of the time.** The worst misses
cluster in two IPL-2018 CSK chases — in one the model had CSK at **0.6%** and they won.
These were extraordinary death-over rescues. A model trained on aggregate history
cannot know a specialist finisher is at the crease; that would require player-level
features.

---

## Match progression

`match_progression()` returns the over-by-over win-probability path for any match, plus
runs and wickets in each over. The notebook plots the IPL-2018 CSK vs MI opener: CSK
bottom out at **0.9%** after over 9, sit under 4% through overs 10–17, then jump to
9.9% and **82.7%** as two 20-run overs land.

That match is kept deliberately rather than cherry-picking one the model handled well —
it shows both that the probability responds sharply and sensibly to each over, and
where the system's ceiling is.

---

## Key cricket insights

| Finding | Value |
|---|---|
| Chasing teams win | **54.25%** of matches |
| RRR ≤ 6 → chase succeeds | **96.5%** |
| RRR > 15 → chase succeeds | **3.2%** |
| 0–2 wickets in hand → win rate | **2.6%** |
| 9–10 wickets in hand → win rate | **65.4%** |
| Target ≤ 140 chased down | **80.6%** |
| Target > 200 chased down | **17.2%** |
| Toss winner wins the match | **51.69%** |

**On the toss:** captains choosing to field win 54.9% vs 46.7% for those choosing to
bat — but this is **correlation, not causation**. Captains field *because* the pitch or
dew already favours chasing, so decision and outcome share a common cause. Toss is not
used as a model feature.

**On balls remaining:** win rate by balls left is almost flat in isolation, because a
team 40 balls in could be cruising or collapsing. Balls only become meaningful in
combination with runs and wickets — which is precisely why the interaction features
exist.

---

## Streamlit application

```bash
streamlit run app.py
```

Loads `models/best_cricket_model.pkl` and `models/model_metadata.json`. Team and city
lists come from the metadata, so the UI can only offer values the model was trained on.

**Inputs:** batting team, bowling team, city, target, current score, wickets lost,
completed overs, balls in current over, and optional recent-momentum values.

**Derived automatically** (the user never enters a redundant value): runs left, balls
left, wickets in hand, CRR, RRR, score progress, and the pressure features.

**Validation blocks impossible states:** identical teams, score at or above target,
120+ balls bowled, 10 wickets down, zero balls bowled, and inconsistent momentum
entries.

**Output:**

```
🏏 Mumbai Indians       — 63%
🏏 Chennai Super Kings  — 37%
```

The two percentages always sum to exactly 100 (one is computed as `100 - other`).
Probabilities come straight from `model.predict_proba()` — nothing hardcoded, nothing
smoothed. An expandable section shows score, runs required, balls remaining, wickets in
hand, CRR and RRR.

---

## A note on `cricket_utils.py`

Both the notebook and the app import this file. It is the single source of truth for
the team mapping, `FEATURE_COLS` (including order), the clipping function, the
over↔ball conversion, and `build_match_state()`.

This exists for a concrete reason: an earlier saved pipeline embedded a
`FunctionTransformer` whose function was not importable, and **failed to unpickle** —
which would have crashed the app on load. Keeping the shared logic in an importable
module fixes that and makes notebook/app drift structurally impossible.

**Keep `cricket_utils.py` next to `app.py` — the saved model depends on it.**

---

## Project structure

```
.
├── cric_ai_pred.ipynb        # full analysis, 120 cells
├── app.py                    # Streamlit application
├── cricket_utils.py          # shared features + preprocessing (single source of truth)
├── requirements.txt
├── matches (1).csv
├── deliveries 2.csv
└── models/
    ├── best_cricket_model.pkl    # preprocessing + model in one pipeline
    └── model_metadata.json       # features, threshold, metrics, teams, cities
```

---

## How to run

```bash
pip install -r requirements.txt

# reproduce the analysis and regenerate the model
jupyter notebook cric_ai_pred.ipynb

# launch the app
streamlit run app.py
```

The notebook contains the code that saves the model — it does not ask you to save
anything manually. Running it top to bottom regenerates both files in `models/`.

**Requires scikit-learn ≥ 1.6** (the calibration section uses
`sklearn.frozen.FrozenEstimator`, which replaced the old `cv='prefit'` argument).

---

## Verification performed

- All 69 code cells execute top to bottom with no errors.
- The tuning cell independently rediscovers the exact hyperparameters used in the
  comparison table — the notebook is internally consistent.
- Saved model reloads; feature order asserted equal to metadata; probabilities sum to
  1.000000.
- Streamlit app boots and serves without error.
- Notebook preprocessing == app preprocessing (same imported functions).
- No forbidden column in `FEATURE_COLS` (asserted, not assumed).
- No `match_id` appears in more than one split (asserted).
- No `126 - (...)` formula and no decimal-over multiplication anywhere in the codebase.

---

## Limitations

1. **Early-chase accuracy is 0.635.** Inherent to the problem — 100 balls is enough
   time for almost anything.
2. **No player-level features.** The model cannot know who is at the crease, which
   causes its worst misses.
3. **Scoring-era drift.** Average targets rose from ~161 in the training seasons to
   ~177 in the test seasons, and the model slightly under-rates chasing teams as a
   result (mean predicted probability 0.475 vs actual win rate 0.568). This is also why
   the tuned threshold sits below 0.50. Production use would need periodic retraining.
4. **Validation season is small** (30 matches), so its scores are noisy and read
   optimistically compared to test. The test figure is the honest headline.
5. **D/L matches excluded** (19 matches).
6. **Data ends at IPL-2019** and covers only the 8 long-running franchises.

## Future improvements

- Batter and bowler identity, with career death-over strike rates
- Venue-specific par scores instead of a plain city one-hot
- Per-season retraining to track scoring drift
- Ball-level rather than over-level momentum windows
