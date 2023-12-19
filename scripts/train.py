# Importing Libraries
print("Importing Libraries")
import json
import pandas as pd
from sklearn.model_selection import train_test_split

from model import evaluate_model, train_model, save_model
from metrics import save_metrics, save_roc_curve
from utils_and_constants import PROCESSED_DATASET, TARGET_COLUMN


def load_data(file_path):
    data = pd.read_csv(file_path)
    X = data.drop(TARGET_COLUMN, axis=1)
    y = data[TARGET_COLUMN]
    return X, y

def main():

    X, y = load_data(PROCESSED_DATASET)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1993)

    model, scaler, vectorizer, metrics = train_model(X_train, y_train)
    metrics, y_proba = evaluate_model(model, scaler,
                                      vectorizer, X_test, y_test)

    print("====================Test Set Metrics==================")
    print(json.dumps(metrics, indent=2))
    print("======================================================")

    save_metrics(metrics)
    save_roc_curve(y_test, y_proba)
    save_model(model, scaler, vectorizer)

if __name__ == "__main__":
    main()