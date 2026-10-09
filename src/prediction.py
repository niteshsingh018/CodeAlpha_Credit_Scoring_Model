import os
import joblib
import pandas as pd


# ==========================================
# PHASE 13: PREDICTION SYSTEM
# ==========================================

print("\n" + "=" * 60)
print("              PHASE 13: PREDICTION SYSTEM")
print("=" * 60)


# ==========================================
# 1. MODEL PATH
# ==========================================

model_file = "models/best_credit_model.pkl"


# ==========================================
# 2. CHECK MODEL
# ==========================================

if not os.path.exists(model_file):

    print("\nERROR: Best model not found!")

    print("Expected file:")
    print(model_file)

    print("\nPlease complete Phase 12 first.")

    exit()


# ==========================================
# 3. LOAD MODEL
# ==========================================

print("\nLoading trained model...")

model = joblib.load(model_file)

print("Model loaded successfully!")


# ==========================================
# 4. LOAD TRAINING FEATURES
# ==========================================

feature_file = "data/X_train.csv"

if not os.path.exists(feature_file):

    print("\nERROR: Training feature file not found!")

    print(f"Expected file: {feature_file}")

    exit()


X_train = pd.read_csv(feature_file)

feature_names = list(X_train.columns)


# ==========================================
# 5. DISPLAY REQUIRED FEATURES
# ==========================================

print("\nModel expects the following features:")

for number, feature in enumerate(feature_names, start=1):

    print(f"{number}. {feature}")


# ==========================================
# 6. GET USER INPUT
# ==========================================

print("\n" + "=" * 60)
print("             CUSTOMER INFORMATION")
print("=" * 60)

print("\nEnter the customer's information.")


def get_number(message):

    while True:

        try:

            value = float(input(message))

            return value

        except ValueError:

            print("Please enter a valid number.")


# ==========================================
# 7. INPUT VALUES
# ==========================================

customer_data = {}


# Numeric features

for feature in feature_names:

    if feature.startswith(
        (
            "occupation_status_",
            "defaults_on_file_",
            "product_type_",
            "loan_intent_"
        )
    ):
        continue

    customer_data[feature] = get_number(
        f"Enter {feature}: "
    )


# ==========================================
# 8. CATEGORICAL / ONE-HOT FEATURES
# ==========================================

print("\nFor categorical features, enter 1 if the category")
print("is applicable, otherwise enter 0.")


for feature in feature_names:

    if feature.startswith(
        (
            "occupation_status_",
            "defaults_on_file_",
            "product_type_",
            "loan_intent_"
        )
    ):

        value = get_number(
            f"Enter {feature} (0 or 1): "
        )

        customer_data[feature] = value


# ==========================================
# 9. CREATE INPUT DATAFRAME
# ==========================================

input_data = pd.DataFrame(
    [customer_data]
)


# ==========================================
# 10. ENSURE CORRECT FEATURE ORDER
# ==========================================

input_data = input_data.reindex(
    columns=feature_names,
    fill_value=0
)


# ==========================================
# 11. MAKE PREDICTION
# ==========================================

print("\n" + "=" * 60)
print("                 PREDICTION")
print("=" * 60)

prediction = model.predict(
    input_data
)[0]


# ==========================================
# 12. GET PROBABILITY
# ==========================================

probability = None

if hasattr(model, "predict_proba"):

    probabilities = model.predict_proba(
        input_data
    )[0]

    probability = max(probabilities)


# ==========================================
# 13. DISPLAY RESULT
# ==========================================

print("\nPrediction completed!")


print("\nPredicted Loan Status:")

print(
    f"Result: {prediction}"
)


if probability is not None:

    print(
        f"Confidence: {probability * 100:.2f}%"
    )


# ==========================================
# 14. CREDIT DECISION
# ==========================================

print("\n" + "-" * 60)

if str(prediction) == "1":

    print("DECISION: LOAN APPROVAL / LOW CREDIT RISK")

else:

    print("DECISION: LOAN REJECTION / HIGH CREDIT RISK")


print("-" * 60)


# ==========================================
# 15. SAVE PREDICTION
# ==========================================

prediction_result = input_data.copy()

prediction_result["Prediction"] = prediction

if probability is not None:

    prediction_result["Confidence"] = probability


output_file = "results/customer_prediction.csv"

prediction_result.to_csv(
    output_file,
    index=False
)


print("\nPrediction saved successfully!")

print(
    f"File: {output_file}"
)


# ==========================================
# 16. FINAL MESSAGE
# ==========================================

print("\n" + "=" * 60)
print("       PHASE 13 COMPLETED SUCCESSFULLY")
print("=" * 60)

print("\nNext Phase:")
print("PHASE 14 - CREDIT SCORING DASHBOARD")