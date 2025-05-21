import json
import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.impute import SimpleImputer

# Configure logging to track all actions in a log file
logging.basicConfig(
    filename="data_cleaning.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


# Function to load data from a JSON file
def load_data(file_path):
    try:
        if file_path.endswith(".json"):
            # Read the raw JSON file content
            with open(file_path, "r") as file:
                raw_data = json.load(file)

            # Try to process if it's a list of dictionaries (tabular data)
            if isinstance(raw_data, list):
                df = pd.DataFrame(raw_data)
                logging.info(f"Data loaded as list of dictionaries from {file_path}")
                print(f"Successfully loaded data from {file_path}")

            # If the JSON is a dictionary with nested fields, try normalizing it
            elif isinstance(raw_data, dict):
                print("Attempting to normalize the nested JSON structure...")
                df = pd.json_normalize(raw_data)
                logging.info(
                    f"Data loaded and normalized from nested dictionary in {file_path}"
                )
                print(f"Successfully loaded and normalized data from {file_path}")

            else:
                raise ValueError(
                    "Unsupported JSON structure. Expected a list or dictionary."
                )

            return df

        else:
            raise ValueError("Unsupported file format. Use JSON.")

    except json.JSONDecodeError as e:
        logging.error(f"Error decoding JSON: {e}")
        print("Error decoding JSON. Please check the file format and content.")
        return None

    except ValueError as e:
        logging.error(f"Error loading data: {e}")
        print(f"Error loading data: {e}")
        return None

    except Exception as e:
        logging.error(f"Error loading data: {e}")
        print(f"Unexpected error occurred while loading data: {e}")
        return None


# Function to inspect and understand the data
def inspect_data(df):
    print("\n--- Data Overview ---")
    print("First 5 rows:")
    print(df.head())
    print("\nColumn Data Types:")
    print(df.dtypes)
    print("\nSummary Statistics:")
    print(df.describe())
    print("\nMissing Values per Column:")
    print(df.isnull().sum())
    # Visualize missing values with a heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.isnull(), cbar=False, cmap="viridis")
    plt.title("Missing Values Heatmap")
    plt.savefig("missing_values.png")
    plt.close()
    logging.info("Data inspection completed")


# Function to handle missing values
def handle_missing_values(df):
    print("\n--- Handling Missing Values ---")
    for column in df.columns:
        missing_count = df[column].isnull().sum()
        if missing_count > 0:
            print(f"Column '{column}' has {missing_count} missing values")
            strategy = input(
                f"Strategy for '{column}' (drop, mean, median, mode, value): "
            ).lower()
            if strategy == "drop":
                df = df.dropna(subset=[column])
                logging.info(f"Dropped rows with missing values in '{column}'")
            elif strategy in ["mean", "median", "mode"]:
                if df[column].dtype in ["float64", "int64"]:
                    imputer = SimpleImputer(strategy=strategy)
                    df[column] = imputer.fit_transform(df[[column]])
                    logging.info(f"Filled '{column}' with {strategy}")
                else:
                    print(f"Cannot use '{strategy}' on non-numeric column")
            elif strategy == "value":
                value = input(f"Enter value to fill in '{column}': ")
                df[column] = df[column].fillna(value)
                logging.info(f"Filled '{column}' with value {value}")
            else:
                print("Invalid strategy, skipping")
    return df


# Function to convert data types
def convert_data_types(df):
    print("\n--- Converting Data Types ---")
    for column in df.columns:
        print(f"Column '{column}' current type: {df[column].dtype}")
        new_type = input(
            f"New type for '{column}' (int, float, str, datetime, skip): "
        ).lower()
        if new_type in ["int", "float", "str", "bool"]:

            try:
                df[column] = df[column].astype(new_type)
                logging.info(f"Converted '{column}' to {new_type}")
            except ValueError as e:
                print(f"Error converting '{column}': {e}")

        elif new_type == "datetime":
            try:
                df[column] = pd.to_datetime(df[column])
                logging.info(f"Converted '{column}' to datetime")
            except Exception as e:
                print(f"Error converting '{column}': {e}")
        elif new_type != "skip":
            print("Invalid type, skipping")
    return df


# Function to remove duplicates
def remove_duplicates(df):
    initial_rows = len(df)
    df = df.drop_duplicates()
    removed = initial_rows - len(df)
    logging.info(f"Removed {removed} duplicate rows")
    print(f"Removed {removed} duplicate rows")
    return df


# Main cleaning function with multiple options
def clean_data(df):
    print("\n--- Cleaning Options ---")
    print("1. Handle missing values")
    print("2. Convert data types")
    print("3. Remove duplicates")
    choices = input("Enter choices (comma-separated, e.g., '1,3') or 'all': ").lower()

    if choices == "all":
        choices = ["1", "2", "3"]
    else:
        choices = choices.split(",")

    for choice in choices:
        if choice == "1":
            df = handle_missing_values(df)
        elif choice == "2":
            df = convert_data_types(df)
        elif choice == "3":
            df = remove_duplicates(df)
        else:
            print(f"Invalid choice: {choice}")
    return df


# Function to save the cleaned data
def save_data(df, output_path):
    try:
        print(df.head())
        if output_path.endswith(".csv"):
            df.to_csv(output_path)
        else:
            raise ValueError("Unsupported format. Use csv.")
        logging.info(f"Data saved to {output_path}")
        print(f"Cleaned data saved to {output_path}")
    except Exception as e:
        logging.error(f"Error saving data: {e}")
        print(f"Error saving data: {e}")


# Main execution block
if __name__ == "__main__":
    # Welcome message and input file path
    print("Welcome to the Smart Data Cleaning Script!")
    print("This script will help you understand, filter, clean, and save your data.")
    input_path = input("Enter path to your input file (JSON): ").strip()

    # Load and process the data
    df = load_data(input_path)
    if df is not None:
        inspect_data(df)
        df = clean_data(df)
        output_path = input("Enter path to save cleaned data (JSON): ").strip()
        save_data(df, output_path)
        print("Data cleaning completed successfully!")
    else:
        print("Failed to load data. Please check the file path and format.")
