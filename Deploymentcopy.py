
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from huggingface_hub import hf_hub_download

model_path = hf_hub_download(repo_id="Yaaba/Training-Model", filename="model.pkl")
with open(model_path, 'rb') as f:
    model = pickle.load(f)


# Main program for Streamlit to use
def main():
    st.title("Player Rating Predictor")
    html_temp = """
    <div style="background:#025246 ;padding:10px">
    <h2 style="color:white;text-align:center;">Sports Prediction App </h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    potential = st.number_input('Player Potential', 1, 100, 1)
    value_eur = st.number_input('Player Value in Euros')
    dribbling = st.number_input('Player Dribbling', 1, 100, 1)
    movement_reactions = st.number_input('Player Reaction')
    international_reputation = st.number_input('Player International Reputation', 1, 5, 1)
    skill_ball_control = st.number_input('Skill Ball Control of The Player')
    goalkeeping_diving = st.number_input('Player Goalkeeping Diving Ability')
    goalkeeping_positioning = st.number_input('Player Goalkeeping Positioning')


    if st.button('Predict'):
        data = {
            'potential': [potential],
            'value_eur': [value_eur],
            'dribbling': [dribbling],
            'movement_reactions': [movement_reactions],
            'international_reputation': [international_reputation],
            'skill_ball_control': [skill_ball_control],
            'goalkeeping_diving': [goalkeeping_diving],
            'goalkeeping_positioning': [goalkeeping_positioning]
        }

        # Making into a DataFrame
        df = pd.DataFrame(data)
        
        # Ensure the DataFrame has the same columns as the model expects
        expected_features = model.feature_names_in_
        for feature in expected_features:
            if feature not in df.columns:
                df[feature] = 0  # or some default value

        df = df[expected_features]  # Reorder columns to match model's expectation
        
        prediction = model.predict(df)
        st.write("The predicted overall for your player is ", prediction[0])

if __name__ == '__main__':
    main()
