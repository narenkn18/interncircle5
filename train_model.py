from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


DATA_URL = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv"
FEATURES = ["pclass", "sex", "age", "fare"]
OUTPUT_PATH = Path(__file__).resolve().parent / "model.joblib"


def load_training_data():
    data = pd.read_csv(DATA_URL)
    data = data[["survived", *FEATURES]].dropna().copy()
    data["sex"] = (data["sex"] == "female").astype(int)
    return data


def train_and_save():
    data = load_training_data()
    X = data[FEATURES]
    y = data["survived"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    depth_results = []
    for depth in range(1, 13):
        candidate = DecisionTreeClassifier(
            max_depth=depth,
            random_state=42,
        )
        candidate.fit(X_train, y_train)
        depth_results.append(
            (
                accuracy_score(y_test, candidate.predict(X_test)),
                depth,
                candidate,
            )
        )

    test_accuracy, best_depth, best_model = max(
        depth_results,
        key=lambda result: result[0],
    )
    joblib.dump(best_model, OUTPUT_PATH)

    print("Saved model to model.joblib")
    print(f"Best maximum depth: {best_depth}")
    print(f"Testing accuracy: {test_accuracy:.2%}")
    print(f"Feature order: {FEATURES}")


if __name__ == "__main__":
    train_and_save()
