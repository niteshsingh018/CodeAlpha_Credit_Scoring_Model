import pandas as pd
import os


# ==========================================
# PHASE 1: DATASET COLLECTION
# ==========================================

print("\n========================================")
print("       PHASE 1: DATASET COLLECTION")
print("========================================")


# Dataset path
dataset_path = "data/Loan_approval_data_2025.csv"


# Check whether dataset exists
if not os.path.exists(dataset_path):
    print("\nERROR: Dataset not found!")
    print("Expected location:")
    print(dataset_path)
    print("\nPlease place Loan_approval_data_2025.csv inside the data folder.")
    exit()


# Load dataset
try:
    df = pd.read_csv(dataset_path)

    print("\nDataset loaded successfully!")


except Exception as e:
    print("\nERROR while loading dataset!")
    print(e)
    exit()


# ==========================================
# DATASET INFORMATION
# ==========================================

print("\n----------------------------------------")
print("Dataset Information")
print("----------------------------------------")

print("Number of rows    :", df.shape[0])
print("Number of columns :", df.shape[1])


# ==========================================
# COLUMN NAMES
# ==========================================

print("\n----------------------------------------")
print("Column Names")
print("----------------------------------------")

for i, column in enumerate(df.columns, start=1):
    print(f"{i}. {column}")


# ==========================================
# FIRST 5 RECORDS
# ==========================================

print("\n----------------------------------------")
print("First 5 Records")
print("----------------------------------------")

print(df.head())


# ==========================================
# DATA TYPES
# ==========================================

print("\n----------------------------------------")
print("Data Types")
print("----------------------------------------")

print(df.dtypes)


# ==========================================
# MISSING VALUES
# ==========================================

print("\n----------------------------------------")
print("Missing Values")
print("----------------------------------------")

missing_values = df.isnull().sum()

print(missing_values)


# ==========================================
# TARGET COLUMN
# ==========================================

target_column = "loan_status"

print("\n----------------------------------------")
print("Target Column")
print("----------------------------------------")

if target_column in df.columns:
    print("Target column found:", target_column)

    print("\nTarget value distribution:")
    print(df[target_column].value_counts())

else:
    print("ERROR: loan_status column not found!")


# ==========================================
# BASIC STATISTICS
# ==========================================

print("\n----------------------------------------")
print("Basic Statistics")
print("----------------------------------------")

print(df.describe(include="all").transpose())


# ==========================================
# FINAL MESSAGE
# ==========================================

print("\n========================================")
print("       PHASE 1 COMPLETED SUCCESSFULLY")
print("========================================")