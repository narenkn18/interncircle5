from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


FEATURES = ["pclass", "sex", "age", "fare"]
MODEL_PATH = Path(__file__).resolve().parent / "model.joblib"


st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="🚢",
    layout="centered",
)


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(MODEL_PATH)
    return joblib.load(MODEL_PATH)


st.title("🚢 Would this passenger have survived?")
st.write(
    "Enter passenger details to get an educational survival estimate "
    "from a Decision Tree model."
)

pclass = st.selectbox(
    "Ticket class",
    options=[1, 2, 3],
    index=1,
)
sex = st.radio(
    "Sex",
    options=["female", "male"],
    horizontal=True,
)
age = st.slider(
    "Age",
    min_value=1,
    max_value=80,
    value=30,
)
fare = st.slider(
    "Fare paid",
    min_value=0.0,
    max_value=500.0,
    value=30.0,
    step=1.0,
)

if st.button("Predict survival", type="primary"):
    try:
        model = load_model()
    except FileNotFoundError:
        st.error(
            "The model file is missing. Run `python train_model.py` "
            "before starting this app."
        )
        st.stop()
    except (OSError, ValueError, EOFError) as error:
        st.error(f"The model could not be loaded: {error}")
        st.stop()

    if not (1 <= pclass <= 3):
        st.error("Ticket class must be 1, 2, or 3.")
        st.stop()
    if not (1 <= age <= 80):
        st.error("Age must be between 1 and 80.")
        st.stop()
    if not (0 <= fare <= 500):
        st.error("Fare must be between 0 and 500.")
        st.stop()

    input_row = pd.DataFrame(
        [[pclass, 1 if sex == "female" else 0, age, fare]],
        columns=FEATURES,
    )

    try:
        probability = model.predict_proba(input_row)[0][1]
    except (ValueError, AttributeError) as error:
        st.error(f"The prediction could not be calculated: {error}")
        st.stop()

    st.metric(
        label="Estimated chance of survival",
        value=f"{probability:.1%}",
    )

    if probability >= 0.5:
        st.success("The model predicts survival.")
    else:
        st.warning("The model predicts not surviving.")

st.divider()
st.caption(
    "This model learns from the historical Titanic passenger dataset. "
    "It is an educational example, not advice or a real-world prediction."
)
