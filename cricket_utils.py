
import numpy as np
import pandas as pd

TEAM_RENAME = {
    'Delhi Daredevils': 'Delhi Capitals',
    'Deccan Chargers': 'Sunrisers Hyderabad',
    'Rising Pune Supergiants': 'Rising Pune Supergiant',  
}
ACTIVE_TEAMS = [
    'Chennai Super Kings',
    'Delhi Capitals',
    'Kings XI Punjab',
    'Kolkata Knight Riders',
    'Mumbai Indians',
    'Rajasthan Royals',
    'Royal Challengers Bangalore',
    'Sunrisers Hyderabad',
]


def normalize_teams(df, columns):
    """Apply the historical franchise renaming to the given team columns."""
    df = df.copy()
    for col in columns:
        df[col] = df[col].replace(TEAM_RENAME)
    return df
CAT_COLS = ['batting_team', 'bowling_team', 'city']
NUM_COLS = [
    'target_score', 'current_score', 'runs_left', 'balls_left', 'wickets',
    'current_run_rate', 'required_run_rate',
    'runs_last_over', 'runs_last_3_overs', 'runs_last_5_overs',
    'wickets_last_over', 'wickets_last_3_overs',
    'score_progress_pct', 'runs_left_per_wicket',
    'rrr_minus_crr', 'pressure_index',
]

FEATURE_COLS = CAT_COLS + NUM_COLS
def clip_extreme_rates(X):
    """Cap mathematically extreme rate features. Used inside the sklearn pipeline."""
    X = pd.DataFrame(X, columns=NUM_COLS).copy()
    X['required_run_rate'] = X['required_run_rate'].clip(upper=36)
    X['pressure_index'] = X['pressure_index'].clip(upper=18)
    return X


def build_preprocessor(scale=False):
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer

    numeric_steps = [('clip', FunctionTransformer(clip_extreme_rates,
                                                  feature_names_out='one-to-one'))]
    if scale:
        numeric_steps.append(('scale', StandardScaler()))

    return ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), CAT_COLS),
        ('num', Pipeline(numeric_steps), NUM_COLS),
    ])
def overs_to_balls(overs_completed, balls_in_current_over):
    return int(overs_completed) * 6 + int(balls_in_current_over)
def balls_to_overs_text(balls):
    return f"{balls // 6}.{balls % 6}"
def build_match_state(batting_team, bowling_team, city, target_score, current_score,
                      balls_completed, wickets_lost,
                      runs_last_over=0, runs_last_3_overs=0, runs_last_5_overs=0,
                      wickets_last_over=0, wickets_last_3_overs=0):
    runs_left = target_score - current_score
    balls_left = 120 - balls_completed
    wickets = 10 - wickets_lost

    current_run_rate = (current_score * 6 / balls_completed) if balls_completed > 0 else 0.0
    required_run_rate = (runs_left * 6 / balls_left) if balls_left > 0 else 0.0
    safe_wickets = wickets if wickets > 0 else 1

    state = {
        'batting_team': batting_team,
        'bowling_team': bowling_team,
        'city': city,
        'target_score': target_score,
        'current_score': current_score,
        'runs_left': runs_left,
        'balls_left': balls_left,
        'wickets': wickets,
        'current_run_rate': current_run_rate,
        'required_run_rate': required_run_rate,
        'runs_last_over': runs_last_over,
        'runs_last_3_overs': runs_last_3_overs,
        'runs_last_5_overs': runs_last_5_overs,
        'wickets_last_over': wickets_last_over,
        'wickets_last_3_overs': wickets_last_3_overs,
        'score_progress_pct': (current_score / target_score * 100) if target_score > 0 else 0.0,
        'runs_left_per_wicket': runs_left / safe_wickets,
        'rrr_minus_crr': required_run_rate - current_run_rate,
        'pressure_index': required_run_rate / (wickets + 1),
    }
    return pd.DataFrame([state])[FEATURE_COLS]
