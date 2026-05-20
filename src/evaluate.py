import pandas as pd
import pickle
import json
import os
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def evaluate():
    # Model load karo
    model = pickle.load(open("models/model.pkl", "rb"))
    
    # Test data load karo
    test_df = pd.read_csv("data/processed/test.csv")
    X_test = test_df.drop("math_score", axis=1)
    y_test = test_df["math_score"]

    # Predict karo
    preds = model.predict(X_test)

    # Metrics calculate karo
    scores = {
        "rmse": round(float(np.sqrt(mean_squared_error(y_test, preds))), 4),
        "mae":  round(float(mean_absolute_error(y_test, preds)), 4),
        "r2":   round(float(r2_score(y_test, preds)), 4)
    }

    os.makedirs("metrics", exist_ok=True)
    json.dump(scores, open("metrics/scores.json", "w"), indent=2)
    
    print(f"✅ Evaluation complete!")
    print(f"RMSE : {scores['rmse']}")
    print(f"MAE  : {scores['mae']}")
    print(f"R²   : {scores['r2']}")

if __name__ == "__main__":
    evaluate()