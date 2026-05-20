# Student Exam Score Predictor 🎓

An end-to-end machine learning pipeline to predict student math scores using DVC for data versioning and experiment tracking.

## Project Overview
- **Problem:** Predict a student's math score based on demographic and academic features
- **Dataset:** Students Performance in Exams (Kaggle) — 1000 students
- **Model:** Random Forest Regressor
- **Best Result:** R² = 0.86, RMSE = 5.819

## Tech Stack
- Python 3.13
- Scikit-learn (Random Forest)
- DVC (Data Version Control)
- AWS S3 (Remote Storage)
- Pandas, NumPy

## Project Structure

student-score-predictor/
├── src/
│   ├── ingest.py        # Stage 1: Data ingestion
│   ├── preprocess.py    # Stage 2: Preprocessing
│   ├── train.py         # Stage 3: Model training
│   └── evaluate.py      # Stage 4: Evaluation
├── data/
│   ├── raw/             # Original dataset
│   └── processed/       # Train/test splits
├── models/              # Trained model
├── metrics/             # Evaluation scores
├── dvc.yaml             # Pipeline definition
├── params.yaml          # Hyperparameters
└── requirements.txt

## Pipeline

Data Ingestion → Preprocessing → Training → Evaluation

Managed by DVC — only changed stages re-run automatically.

## Results

| Metric | Score |
|--------|-------|
| R²     | 0.861 |
| RMSE   | 5.819 |
| MAE    | 4.517 |

## Experiment Tracking

| Experiment | n_estimators | max_depth | R²    |
|------------|-------------|-----------|-------|
| baseline   | 100         | 6         | 0.861 ← Best |
| exp-2      | 200         | 12        | 0.850 |
| exp-3      | 50          | 3         | 0.801 |

## How to Run

### 1. Clone the repo
```bash
git clone <your-repo-url>
cd student-score-predictor
```

### 2. Install dependencies
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Pull data from S3
```bash
dvc pull
```

### 4. Run full pipeline
```bash
dvc repro
```

### 5. Check metrics
```bash
dvc metrics show
```

### 6. Run experiments
```bash
dvc exp run --set-param train.n_estimators=200
dvc exp show
```

## Key Learnings
- Built reproducible ML pipeline using DVC
- Tracked data versions with MD5 hashing
- Managed hyperparameters via params.yaml
- Compared multiple experiments systematically
- Used AWS S3 as remote artifact storage