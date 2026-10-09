
import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# ==========================================
# PHASE 10: MODEL EVALUATION
# ==========================================

print("\n" + "=" * 60)
print("          PHASE 10: MODEL EVALUATION")
print("=" * 60)

os.makedirs("results", exist_ok=True)

evaluation_file = "results/model_evaluation.csv"

model_files = {
    "Logistic Regression": "results/logistic_predictions.csv",
    "Decision Tree": "results/decision_tree_predictions.csv",
    "Random Forest": "results/random_forest_predictions.csv"
}


# ==========================================
# EVALUATE ONE MODEL
# ==========================================

def evaluate_model(model_name, file_path):

    print(f"\nEvaluating: {model_name}")

    if not os.path.exists(file_path):
        print(f"ERROR: File not found: {file_path}")
        print("Run the corresponding model training script first.")
        return None

    df = pd.read_csv(file_path)

    required_columns = ["Actual", "Predicted", "Probability"]

    for column in required_columns:
        if column not in df.columns:
            print(f"ERROR: Missing column '{column}'")
            print("Available columns:", df.columns.tolist())
            return None

    # Remove rows with missing values in required columns
    df = df[required_columns].dropna()

    if df.empty:
        print(f"ERROR: No usable predictions for {model_name}")
        return None

    y_true = df["Actual"]
    y_pred = df["Predicted"]
    y_prob = df["Probability"]

    results = {
        "Model": model_name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(
            y_true, y_pred, average="binary", zero_division=0
        ),
        "Recall": recall_score(
            y_true, y_pred, average="binary", zero_division=0
        ),
        "F1-Score": f1_score(
            y_true, y_pred, average="binary", zero_division=0
        )
    }

    # ROC-AUC requires both classes to be present
    if y_true.nunique() == 2:
        results["ROC-AUC"] = roc_auc_score(y_true, y_prob)
    else:
        results["ROC-AUC"] = float("nan")

    # Create confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    display = ConfusionMatrixDisplay(confusion_matrix=cm)
    display.plot()

    safe_name = model_name.lower().replace(" ", "_")
    image_path = f"results/{safe_name}_confusion_matrix.png"

    plt.title(f"{model_name} Confusion Matrix")
    plt.tight_layout()
    plt.savefig(image_path)
    plt.close()

    print(f"Accuracy: {results['Accuracy']:.4f}")
    print(f"F1-Score: {results['F1-Score']:.4f}")
    print(f"ROC-AUC: {results['ROC-AUC']:.4f}")
    print(f"Confusion matrix saved: {image_path}")

    return results


# ==========================================
# EVALUATE ALL THREE MODELS
# ==========================================

all_results = []

for model_name, file_path in model_files.items():

    result = evaluate_model(model_name, file_path)

    if result is not None:
        all_results.append(result)


# ==========================================
# CHECK RESULTS
# ==========================================

if not all_results:
    print("\nERROR: No models were evaluated.")
    print("Check that all three prediction CSV files exist.")
    raise SystemExit(1)


# ==========================================
# CREATE AND SAVE RESULTS
# ==========================================

results = pd.DataFrame(all_results)

metrics = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1-Score",
    "ROC-AUC"
]

results[metrics] = results[metrics].round(4)

print("\n" + "=" * 60)
print("             MODEL EVALUATION RESULTS")
print("=" * 60)

print(results.to_string(index=False))

results.to_csv(evaluation_file, index=False)

print(f"\nEvaluation results saved: {evaluation_file}")


# ==========================================
# FIND BEST MODEL
# ==========================================

valid_results = results.dropna(subset=["ROC-AUC"])

if not valid_results.empty:

    best_model = valid_results.loc[
        valid_results["ROC-AUC"].idxmax()
    ]

    print("\n" + "=" * 60)
    print("                 BEST MODEL")
    print("=" * 60)

    print(f"Model    : {best_model['Model']}")
    print(f"Accuracy : {best_model['Accuracy']}")
    print(f"F1-Score : {best_model['F1-Score']}")
    print(f"ROC-AUC  : {best_model['ROC-AUC']}")

else:
    print("\nWARNING: Cannot select the best model by ROC-AUC.")


# ==========================================
# COMPLETION
# ==========================================

print("\n" + "=" * 60)
print("       PHASE 10 COMPLETED SUCCESSFULLY")
print("=" * 60)

print("\nNext Phase: PHASE 11 - MODEL COMPARISON")

