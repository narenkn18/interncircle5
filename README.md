# Titanic Survival Predictor

A Streamlit web app that predicts Titanic passenger survival with a saved
Decision Tree classifier.

## Run locally

```bash
python -m pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

Open the local URL shown by Streamlit.

## Model inputs

The model uses features in this exact order:

1. `pclass`
2. `sex`
3. `age`
4. `fare`

`sex` is encoded as `1` for female and `0` for male.

## Files

- `app.py` - Streamlit interface and prediction flow
- `train_model.py` - downloads data, trains candidates, and saves the best model
- `model.joblib` - saved trained model
- `requirements.txt` - deployment dependencies

## Streamlit Community Cloud

Push this folder to a public GitHub repository and create a Streamlit app from
`app.py`. The historical Titanic dataset is downloaded only while retraining;
the deployed app loads the bundled `model.joblib`.

This is an educational project and is not advice.
