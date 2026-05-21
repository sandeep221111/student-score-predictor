import pandas as pd
import pickle
import yaml
import os
import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def train():
    params = yaml.safe_load(open("params.yaml"))["train"]
    
    train_df = pd.read_csv("data/processed/train.csv")
    test_df = pd.read_csv("data/processed/test.csv")
    X_train = train_df.drop("math_score", axis=1)
    y_train = train_df["math_score"]
    X_test = test_df.drop("math_score", axis=1)
    y_test = test_df["math_score"]

    mlflow.set_tracking_uri("sqlite:///mlruns.db")
    mlflow.set_experiment("student-score-predictor")

    with mlflow.start_run():
        model = RandomForestRegressor(
            n_estimators = params["n_estimators"],
            max_depth    = params["max_depth"],
            random_state = params["random_state"]
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        mlflow.log_param("n_estimators", params["n_estimators"])
        mlflow.log_param("max_depth", params["max_depth"])
        mlflow.log_param("random_state", params["random_state"])

        rmse = round(float(np.sqrt(mean_squared_error(y_test, preds))), 4)
        mae  = round(float(mean_absolute_error(y_test, preds)), 4)
        r2   = round(float(r2_score(y_test, preds)), 4)

        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)

        os.makedirs("models", exist_ok=True)
        pickle.dump(model, open("models/model.pkl", "wb"))
        mlflow.sklearn.log_model(model, "random-forest-model")

        print(f"✅ Model trained!")
        print(f"RMSE: {rmse} | MAE: {mae} | R²: {r2}")

if __name__ == "__main__":
    train()
