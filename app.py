
import json
import joblib
import pandas as pd
import streamlit as st
from cricket_utils import build_match_state, balls_to_overs_text
MODEL_PATH = "models/best_cricket_model.pkl"
METADATA_PATH = "models/model_metadata.json"
st.set_page_config(page_title="IPL Win Probability Predictor", page_icon="🏏",
                   layout="centered")

@st.cache_resource
def load_model_and_metadata():
    model = joblib.load(MODEL_PATH)
    with open(METADATA_PATH) as f:
        metadata = json.load(f)
    return model, metadata
model, metadata = load_model_and_metadata()
teams = metadata["teams"]
cities = metadata["cities"]
st.title("🏏 IPL Win Probability Predictor")
st.caption(
    f"Second-innings win probability from the current match situation. "
    f"Model: {metadata['model_name']} | test ROC-AUC {metadata['test_metrics']['ROC-AUC']} "
    f"on {metadata['n_test_matches']} unseen matches ({', '.join(metadata['test_seasons'])})."
)
st.subheader("Match setup")
col1, col2 = st.columns(2)
with col1:
    batting_team = st.selectbox("Batting team (chasing)", teams, index=teams.index("Mumbai Indians"))
with col2:
    bowling_team = st.selectbox("Bowling team (defending)", teams,
                                index=teams.index("Chennai Super Kings"))
city = st.selectbox("Host city", cities)
target_score = st.number_input("Target score", min_value=1, max_value=300, value=180, step=1)
st.subheader("Current situation")
col3, col4 = st.columns(2)
with col3:
    current_score = st.number_input("Current score", min_value=0, max_value=299, value=120, step=1)
    wickets_lost = st.number_input("Wickets lost", min_value=0, max_value=10, value=4, step=1)
with col4:
    overs_done = st.number_input("Completed overs", min_value=0, max_value=19, value=15, step=1)
    balls_this_over = st.number_input("Balls in current over", min_value=0, max_value=5,
                                      value=0, step=1)
balls_completed = overs_done * 6 + balls_this_over
with st.expander("Recent momentum (optional)"):
    st.caption("Leave at 0 if you don't have these to hand - the model still works, "
               "but momentum features are what the model saw during training.")
    mcol1, mcol2, mcol3 = st.columns(3)
    with mcol1:
        runs_last_over = st.number_input("Runs in last over", 0, 40, 0)
        wickets_last_over = st.number_input("Wickets in last over", 0, 3, 0)
    with mcol2:
        runs_last_3_overs = st.number_input("Runs in last 3 overs", 0, 100, 0)
        wickets_last_3_overs = st.number_input("Wickets in last 3 overs", 0, 6, 0)
    with mcol3:
        runs_last_5_overs = st.number_input("Runs in last 5 overs", 0, 150, 0)

errors = []
if batting_team == bowling_team:
    errors.append("Batting and bowling teams must be different.")
if current_score >= target_score:
    errors.append("The chase is already complete - current score must be below the target.")
if balls_completed >= 120:
    errors.append("An innings is 120 legal balls (20 overs); the innings is already over.")
if wickets_lost >= 10:
    errors.append("All 10 wickets have fallen - the innings is already over.")
if balls_completed == 0:
    errors.append("Enter at least one ball bowled so a current run rate can be computed.")
if runs_last_over > runs_last_3_overs > 0:
    errors.append("Runs in the last over cannot exceed runs in the last 3 overs.")
if runs_last_3_overs > runs_last_5_overs > 0:
    errors.append("Runs in the last 3 overs cannot exceed runs in the last 5 overs.")
if wickets_last_over + wickets_last_3_overs > wickets_lost:
    errors.append("Recent wickets cannot exceed total wickets lost.")

if st.button("Predict win probability", type="primary"):
    if errors:
        for e in errors:
            st.error(e)
    else:
        match_state = build_match_state(
            batting_team=batting_team,
            bowling_team=bowling_team,
            city=city,
            target_score=target_score,
            current_score=current_score,
            balls_completed=balls_completed,
            wickets_lost=wickets_lost,
            runs_last_over=runs_last_over,
            runs_last_3_overs=runs_last_3_overs,
            runs_last_5_overs=runs_last_5_overs,
            wickets_last_over=wickets_last_over,
            wickets_last_3_overs=wickets_last_3_overs,
        )

        probabilities = model.predict_proba(match_state)[0]
        lose_probability = probabilities[0]
        win_probability = probabilities[1]
        batting_pct = round(win_probability * 100)
        bowling_pct = 100 - batting_pct

        st.markdown("---")
        result_col1, result_col2 = st.columns(2)
        result_col1.metric(f"🏏 {batting_team}", f"{batting_pct}%")
        result_col2.metric(f"🏏 {bowling_team}", f"{bowling_pct}%")
        st.progress(win_probability)

        if win_probability >= 0.65:
            st.success(f"{batting_team} are in a strong position.")
        elif win_probability >= 0.35:
            st.info("Competitive position - the chase is finely balanced.")
        else:
            st.warning(f"{batting_team} are under pressure.")

        with st.expander("Match situation details"):
            state = match_state.iloc[0]
            details = pd.DataFrame({
                "Metric": ["Target", "Current score", "Runs required", "Balls remaining",
                           "Overs bowled", "Wickets in hand",
                           "Current run rate", "Required run rate"],
                "Value": [target_score, current_score, int(state["runs_left"]),
                          int(state["balls_left"]), balls_to_overs_text(balls_completed),
                          int(state["wickets"]),
                          f"{state['current_run_rate']:.2f}",
                          f"{state['required_run_rate']:.2f}"],
            })
            st.dataframe(details, hide_index=True, width="stretch")

st.markdown("---")
st.caption(
    "Trained on IPL 2008-2016, validated on 2017 and tested on 2018-2019. "
    "Probabilities come directly from the model's predict_proba - no manual adjustment."
)
