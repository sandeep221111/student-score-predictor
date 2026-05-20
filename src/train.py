import pandas as pd
import pickle
import yaml
import os
from sklearn.ensemble import RandomForestRegressor

def train():
    # Params load karo
    params = yaml.safe_load(open("params.yaml"))["train"]
    print(f"Params: {params}")

    # Data load karo
    train_df = pd.read_csv("data/processed/train.csv")
    X_train = train_df.drop("math_score", axis=1)
    y_train = train_df["math_score"]

    # Model banao
    model = RandomForestRegressor(
        n_estimators = params["n_estimators"],
        max_depth    = params["max_depth"],
        random_state = params["random_state"]
    )
    model.fit(X_train, y_train)

    # Model save karo
    os.makedirs("models", exist_ok=True)
    pickle.dump(model, open("models/model.pkl", "wb"))
    print(f"✅ Model trained & saved!")

if __name__ == "__main__":
    train()