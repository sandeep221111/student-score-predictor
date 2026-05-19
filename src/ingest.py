import pandas as pd
import os

def ingest():
    os.makedirs("data/raw", exist_ok=True)
    
    # Kaggle se downloaded file read karo
    df = pd.read_csv("data/raw/StudentsPerformance.csv")
    
    # Column names clean karo
    df.columns = [c.strip().lower().replace(" ", "_").replace("/", "_") 
                  for c in df.columns]
    
    # Clean file save karo
    df.to_csv("data/raw/students.csv", index=False)
    
    print(f"✅ Dataset ready: {len(df)} rows, {len(df.columns)} columns")
    print(f"\nColumns: {list(df.columns)}")
    print(f"\nFirst 3 rows:")
    print(df.head(3))

if __name__ == "__main__":
    ingest()