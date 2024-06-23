
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from huggingface_hub import hf_hub_download

model_path = hf_hub_download(repo_id="Yaaba/Training-Model", filename="model.pkl")
with open(model_path, 'rb') as f:
    model = pickle.load(f)

# Load the pre-trained model
model = pickle.load(open('model.pkl', 'rb'))


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
        # df = pd.DataFrame(data)
        df = pd.DataFrame(data, columns=model.feature_names_in_)
        prediction = model.predict(df)
        st.write("The predicted overall for your player is ", prediction[0])

if __name__ == '__main__':
    main()
