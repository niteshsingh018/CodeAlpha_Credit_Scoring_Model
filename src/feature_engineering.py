import pandas as pd
import os


# ==========================================
# PHASE 4: FEATURE ENGINEERING
# ==========================================

print("\n========================================")
print("       PHASE 4: FEATURE ENGINEERING")
print("========================================")


# ==========================================
# FILE PATHS
# ==========================================

input_file = "data/credit_risk_cleaned.csv"
output_file = "data/credit_risk_features.csv"


# ==========================================
# CHECK INPUT FILE
# ==========================================

if not os.path.exists(input_file):

    print("\nERROR: Cleaned dataset not found!")

    print("Expected file:")
    print(input_file)

    print("\nPlease run Phase 3 first:")
    print("python src\\cleaning.py")

    exit()


# ==========================================
# LOAD DATASET
# ==========================================

print("\nLoading cleaned dataset...")

df = pd.read_csv(input_file)

print("Dataset loaded successfully!")

print("\nOriginal shape:")
print("Rows    :", df.shape[0])
print("Columns :", df.shape[1])


# ==========================================
# CREATE FEATURE: INCOME TO LOAN RATIO
# ==========================================

print("\n----------------------------------------")
print("1. CREATING FINANCIAL FEATURES")
print("----------------------------------------")


if "annual_income" in df.columns and "loan_amount" in df.columns:

    df["income_to_loan_ratio"] = (
        df["annual_income"] /
        (df["loan_amount"] + 1)
    )

    print("Created: income_to_loan_ratio")


# ==========================================
# INCOME TO DEBT RATIO
# ==========================================

if "annual_income" in df.columns and "current_debt" in df.columns:

    df["income_to_debt_ratio"] = (
        df["annual_income"] /
        (df["current_debt"] + 1)
    )

    print("Created: income_to_debt_ratio")


# ==========================================
# SAVINGS TO DEBT RATIO
# ==========================================

if "savings_assets" in df.columns and "current_debt" in df.columns:

    df["savings_to_debt_ratio"] = (
        df["savings_assets"] /
        (df["current_debt"] + 1)
    )

    print("Created: savings_to_debt_ratio")


# ==========================================
# CREDIT HISTORY TO AGE RATIO
# ==========================================

if "credit_history_years" in df.columns and "age" in df.columns:

    df["credit_history_to_age_ratio"] = (
        df["credit_history_years"] /
        (df["age"] + 1)
    )

    print("Created: credit_history_to_age_ratio")


# ==========================================
# EMPLOYMENT TO AGE RATIO
# ==========================================

if "years_employed" in df.columns and "age" in df.columns:

    df["employment_to_age_ratio"] = (
        df["years_employed"] /
        (df["age"] + 1)
    )

    print("Created: employment_to_age_ratio")


# ==========================================
# TOTAL DEBT BURDEN
# ==========================================

if (
    "current_debt" in df.columns
    and "loan_amount" in df.columns
):

    df["total_debt_burden"] = (
        df["current_debt"] +
        df["loan_amount"]
    )

    print("Created: total_debt_burden")


# ==========================================
# FINANCIAL STRESS RATIO
# ==========================================

if (
    "current_debt" in df.columns
    and "loan_amount" in df.columns
    and "annual_income" in df.columns
):

    df["financial_stress_ratio"] = (
        (df["current_debt"] + df["loan_amount"]) /
        (df["annual_income"] + 1)
    )

    print("Created: financial_stress_ratio")


# ==========================================
# HANDLE INFINITE VALUES
# ==========================================

print("\n----------------------------------------")
print("2. HANDLING INFINITE VALUES")
print("----------------------------------------")

df = df.replace(
    [float("inf"), float("-inf")],
    pd.NA
)

print("Infinite values replaced.")


# ==========================================
# HANDLE NEW FEATURE MISSING VALUES
# ==========================================

df = df.fillna(0)

print("Missing feature values handled.")


# ==========================================
# ENCODE CATEGORICAL VARIABLES
# ==========================================

print("\n----------------------------------------")
print("3. ENCODING CATEGORICAL VARIABLES")
print("----------------------------------------")


# Columns that should NOT be used as model features
# customer_id is only an identifier.

categorical_columns = [
    "occupation_status",
    "defaults_on_file",
    "product_type",
    "loan_intent"
]


existing_categorical_columns = [
    column
    for column in categorical_columns
    if column in df.columns
]


if len(existing_categorical_columns) > 0:

    print("\nCategorical columns found:")

    for column in existing_categorical_columns:
        print("-", column)

    df = pd.get_dummies(
        df,
        columns=existing_categorical_columns,
        drop_first=True,
        dtype=int
    )

    print("\nCategorical variables encoded successfully.")

else:

    print("No expected categorical columns found.")


# ==========================================
# CONVERT BOOLEAN COLUMNS
# ==========================================

print("\n----------------------------------------")
print("4. CHECKING DATA TYPES")
print("----------------------------------------")


for column in df.columns:

    if df[column].dtype == "bool":

        df[column] = df[column].astype(int)


print("Boolean values converted to integers.")


# ==========================================
# REMOVE CUSTOMER ID
# ==========================================

print("\n----------------------------------------")
print("5. REMOVING IDENTIFIER COLUMN")
print("----------------------------------------")


if "customer_id" in df.columns:

    df = df.drop(
        columns=["customer_id"]
    )

    print("Removed: customer_id")

else:

    print("customer_id already removed.")


# ==========================================
# CHECK TARGET
# ==========================================

print("\n----------------------------------------")
print("6. CHECKING TARGET COLUMN")
print("----------------------------------------")


if "loan_status" not in df.columns:

    print("ERROR: loan_status not found!")

    exit()


print("Target column found: loan_status")


# ==========================================
# DISPLAY NEW FEATURES
# ==========================================

print("\n----------------------------------------")
print("7. FEATURE LIST")
print("----------------------------------------")

for i, column in enumerate(df.columns, 1):

    print(f"{i}. {column}")


# ==========================================
# SAVE FEATURE DATASET
# ==========================================

print("\n----------------------------------------")
print("8. SAVING FEATURE DATASET")
print("----------------------------------------")


os.makedirs("data", exist_ok=True)


df.to_csv(
    output_file,
    index=False
)


print("\nFeature dataset saved successfully!")

print("File:")
print(output_file)


# ==========================================
# FINAL INFORMATION
# ==========================================

print("\n----------------------------------------")
print("9. FINAL DATASET INFORMATION")
print("----------------------------------------")

print("Rows    :", df.shape[0])
print("Columns :", df.shape[1])


# ==========================================
# FINAL MESSAGE
# ==========================================

print("\n========================================")
print("     PHASE 4 COMPLETED SUCCESSFULLY")
print("========================================")

print("\nCreated file:")
print("data/credit_risk_features.csv")

print("\nNext phase:")
print("Phase 5 - Prepare X and y")