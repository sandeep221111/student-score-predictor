import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os

def preprocess():
    df = pd.read_csv("data/raw/students.csv")
    print(f"Raw data shape: {df.shape}")

    # Categorical columns encode karo (string → number)
    cat_cols = ["gender", "race_ethnicity", "parental_level_of_education",
                "lunch", "test_preparation_course"]
    
    le = LabelEncoder()
    for col in cat_cols:
        df[col] = le.fit_transform(df[col])

    # Features aur target alag karo
    X = df.drop("math_score", axis=1)
    y = df["math_score"]

    # 80% train, 20% test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    os.makedirs("data/processed", exist_ok=True)
    
    train = X_train.copy()
    train["math_score"] = y_train
    test = X_test.copy()
    test["math_score"] = y_test

    train.to_csv("data/processed/train.csv", index=False)
    test.to_csv("data/processed/test.csv", index=False)
    
    print(f"✅ Train: {len(train)} rows | Test: {len(test)} rows")
    print(f"Features: {list(X.columns)}")

if __name__ == "__main__":
    preprocess()