# CrediScope — Creditworthiness Prediction System

A Python machine-learning project with a Streamlit web app for reviewing the dataset, comparing classification models, and testing predictions using the saved model.

## Features

- Overview dashboard with dataset and model summaries
- Applicant prediction form
- Model evaluation and comparison pages
- Dataset insights and preview
- Logistic Regression, Decision Tree, and Random Forest artifacts
- Existing training scripts under `src/`

## Project structure

```text
credit_scoring_model/
├── app.py
├── requirements.txt
├── README.md
├── data/
├── models/
├── results/
└── src/
```

## Run on Windows (PowerShell)

1. Extract this project ZIP.
2. Open PowerShell in the `credit_scoring_model` folder (the folder containing `app.py`).
3. Create and activate a virtual environment:

   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

   If PowerShell blocks activation for this session, run:

   ```powershell
   Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
   .\.venv\Scripts\Activate.ps1
   ```

4. Install dependencies:

   ```powershell
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

5. Start the web app:

   ```powershell
   python -m streamlit run app.py
   ```

6. Open the local URL printed in the terminal (usually `http://localhost:8501`).

## Existing model/data files

The app expects these paths relative to the project folder:

- `models/best_credit_model.pkl`
- `data/X_train.csv` (feature order)
- `data/credit_risk_features.csv`
- `data/credit_risk_cleaned.csv`
- `results/model_evaluation.csv`
- `results/model_comparison.csv`

Do not rename these folders unless you also update the paths in `app.py`.

## Before submission

- Run the app and test each navigation page.
- Test at least one prediction and verify the generated 31 features match the training columns.
- Ensure the evaluation and comparison CSV files contain all three classifiers and are consistent with the model selected as `best_credit_model.pkl`.
- Include screenshots of the dashboard, prediction result, and model comparison in your report.
- Document the source of the dataset and the exact meaning of the target labels.

## Academic-use notice

The predictions are outputs of a trained model, not financial advice or a real lending decision. Do not use this project to make decisions about real applicants without proper validation, fairness testing, security, and compliance review.
