import pandas as pd
import os


# ==========================================
# PHASE 3: DATA CLEANING
# ==========================================

print("\n========================================")
print("          PHASE 3: DATA CLEANING")
print("========================================")


# ==========================================
# FILE PATHS
# ==========================================

input_file = "data/Loan_approval_data_2025.csv"
output_file = "data/credit_risk_cleaned.csv"


# ==========================================
# CHECK INPUT FILE
# ==========================================

if not os.path.exists(input_file):

    print("\nERROR: Dataset not found!")
    print("Expected file:")
    print(input_file)

    exit()


# ==========================================
# LOAD DATA
# ==========================================

print("\nLoading dataset...")

df = pd.read_csv(input_file)

print("Dataset loaded successfully!")

print("\nOriginal dataset shape:")
print("Rows    :", df.shape[0])
print("Columns :", df.shape[1])


# ==========================================
# REMOVE DUPLICATES
# ==========================================

print("\n----------------------------------------")
print("1. REMOVING DUPLICATES")
print("----------------------------------------")

duplicate_count = df.duplicated().sum()

print("Duplicates found:", duplicate_count)

if duplicate_count > 0:

    df = df.drop_duplicates()

    print("Duplicates removed.")

else:

    print("No duplicate records found.")


# ==========================================
# HANDLE MISSING VALUES
# ==========================================

print("\n----------------------------------------")
print("2. HANDLING MISSING VALUES")
print("----------------------------------------")

missing_before = df.isnull().sum()

print("\nMissing values before cleaning:")

print(
    missing_before[missing_before > 0]
    .to_string()
    if missing_before.sum() > 0
    else "No missing values found."
)


# Numerical columns
numerical_columns = df.select_dtypes(
    include=["int64", "float64"]
).columns


# Categorical columns
categorical_columns = df.select_dtypes(
    include=["object"]
).columns


# Fill numerical missing values with median
for column in numerical_columns:

    if df[column].isnull().sum() > 0:

        median_value = df[column].median()

        df[column] = df[column].fillna(median_value)

        print(
            f"Filled missing values in {column} "
            f"with median: {median_value}"
        )


# Fill categorical missing values with mode
for column in categorical_columns:

    if df[column].isnull().sum() > 0:

        mode_value = df[column].mode()[0]

        df[column] = df[column].fillna(mode_value)

        print(
            f"Filled missing values in {column} "
            f"with mode: {mode_value}"
        )


# ==========================================
# CHECK TARGET COLUMN
# ==========================================

print("\n----------------------------------------")
print("3. CHECKING TARGET COLUMN")
print("----------------------------------------")

target_column = "loan_status"


if target_column not in df.columns:

    print("\nERROR: loan_status column not found!")

    print("\nAvailable columns:")

    for column in df.columns:
        print("-", column)

    exit()


print("Target column found:", target_column)


# Remove rows where target is missing
target_missing = df[target_column].isnull().sum()

if target_missing > 0:

    print(
        "Removing",
        target_missing,
        "records with missing target values."
    )

    df = df.dropna(subset=[target_column])

else:

    print("No missing target values.")


# ==========================================
# CHECK NUMERICAL VALUES
# ==========================================

print("\n----------------------------------------")
print("4. CHECKING NUMERICAL VALUES")
print("----------------------------------------")


# Age
if "age" in df.columns:

    invalid_age = (
        (df["age"] < 18) |
        (df["age"] > 100)
    ).sum()

    print("Invalid age records:", invalid_age)

    if invalid_age > 0:

        df = df[
            (df["age"] >= 18) &
            (df["age"] <= 100)
        ]


# Annual income
if "annual_income" in df.columns:

    invalid_income = (
        df["annual_income"] <= 0
    ).sum()

    print("Invalid annual income records:", invalid_income)

    if invalid_income > 0:

        df = df[
            df["annual_income"] > 0
        ]


# Credit score
if "credit_score" in df.columns:

    invalid_credit_score = (
        (df["credit_score"] < 0) |
        (df["credit_score"] > 1000)
    ).sum()

    print(
        "Invalid credit score records:",
        invalid_credit_score
    )

    if invalid_credit_score > 0:

        df = df[
            (df["credit_score"] >= 0) &
            (df["credit_score"] <= 1000)
        ]


# Years employed
if "years_employed" in df.columns:

    invalid_employment = (
        df["years_employed"] < 0
    ).sum()

    print(
        "Invalid years employed records:",
        invalid_employment
    )

    if invalid_employment > 0:

        df = df[
            df["years_employed"] >= 0
        ]


# Credit history
if "credit_history_years" in df.columns:

    invalid_history = (
        df["credit_history_years"] < 0
    ).sum()

    print(
        "Invalid credit history records:",
        invalid_history
    )

    if invalid_history > 0:

        df = df[
            df["credit_history_years"] >= 0
        ]


# Savings
if "savings_assets" in df.columns:

    invalid_savings = (
        df["savings_assets"] < 0
    ).sum()

    print(
        "Invalid savings records:",
        invalid_savings
    )

    if invalid_savings > 0:

        df = df[
            df["savings_assets"] >= 0
        ]


# Current debt
if "current_debt" in df.columns:

    invalid_debt = (
        df["current_debt"] < 0
    ).sum()

    print(
        "Invalid current debt records:",
        invalid_debt
    )

    if invalid_debt > 0:

        df = df[
            df["current_debt"] >= 0
        ]


# Loan amount
if "loan_amount" in df.columns:

    invalid_loan = (
        df["loan_amount"] <= 0
    ).sum()

    print(
        "Invalid loan amount records:",
        invalid_loan
    )

    if invalid_loan > 0:

        df = df[
            df["loan_amount"] > 0
        ]


# ==========================================
# RESET INDEX
# ==========================================

df = df.reset_index(drop=True)


# ==========================================
# FINAL MISSING VALUE CHECK
# ==========================================

print("\n----------------------------------------")
print("5. FINAL MISSING VALUE CHECK")
print("----------------------------------------")

remaining_missing = df.isnull().sum().sum()

print(
    "Total remaining missing values:",
    remaining_missing
)


# ==========================================
# FINAL DUPLICATE CHECK
# ==========================================

print("\n----------------------------------------")
print("6. FINAL DUPLICATE CHECK")
print("----------------------------------------")

remaining_duplicates = df.duplicated().sum()

print(
    "Remaining duplicate records:",
    remaining_duplicates
)


# ==========================================
# SAVE CLEANED DATASET
# ==========================================

print("\n----------------------------------------")
print("7. SAVING CLEANED DATASET")
print("----------------------------------------")


# Create data folder if necessary
os.makedirs("data", exist_ok=True)


df.to_csv(
    output_file,
    index=False
)


print("\nCleaned dataset saved successfully!")

print("File:", output_file)


# ==========================================
# FINAL DATASET INFORMATION
# ==========================================

print("\n----------------------------------------")
print("8. CLEANED DATASET INFORMATION")
print("----------------------------------------")

print("Rows    :", df.shape[0])
print("Columns :", df.shape[1])


# ==========================================
# TARGET DISTRIBUTION
# ==========================================

print("\n----------------------------------------")
print("9. LOAN STATUS DISTRIBUTION")
print("----------------------------------------")

print(
    df["loan_status"]
    .value_counts()
    .to_string()
)


# ==========================================
# FINAL MESSAGE
# ==========================================

print("\n========================================")
print("       PHASE 3 COMPLETED SUCCESSFULLY")
print("========================================")

print("\nOutput file:")
print("data/credit_risk_cleaned.csv")

print("\nNext phase:")
print("Phase 4 - Feature Engineering")