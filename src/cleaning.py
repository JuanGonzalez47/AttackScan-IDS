# cleaning.py
# Data cleaning pipeline extracted from 02_data_cleaning.ipynb

import os
import pandas as pd

def clean_data(output_path):
    """
    Load raw CSVs, merge, clean duplicates, validate types, inspect missing values, and export cleaned dataset.
    """
    bronze_files = [
        'data/bronze/Benign Traffic.csv',
        'data/bronze/DDoS ICMP Flood.csv',
        'data/bronze/DDoS UDP Flood.csv',
        'data/bronze/DoS ICMP Flood.csv',
        'data/bronze/DoS TCP Flood.csv',
        'data/bronze/DoS UDP Flood.csv',
        'data/bronze/MITM ARP Spoofing.csv',
        'data/bronze/MQTT DDoS Publish Flood.csv',
        'data/bronze/MQTT DoS Connect Flood.csv',
        'data/bronze/MQTT DoS Publish Flood.csv',
        'data/bronze/MQTT Malformed.csv',
        'data/bronze/Recon OS Scan.csv',
        'data/bronze/Recon Ping Sweep.csv',
        'data/bronze/Recon Port Scan.csv',
        'data/bronze/Recon Vulnerability Scan.csv',
    ]
    dfs = [pd.read_csv(f) for f in bronze_files]
    df_all = pd.concat(dfs, ignore_index=True)
    # Remove full duplicates
    df_all = df_all.drop_duplicates().reset_index(drop=True)
    # Inspect missing values
    print("Null values per column:")
    print(df_all.isnull().sum())
    # --- Data type standardization ---
    print("Standardizing data types...\n")
    # 1. Convert Timestamp to datetime
    if 'Timestamp' in df_all.columns:
        print("Converting Timestamp to datetime...")
        df_all['Timestamp'] = pd.to_datetime(df_all['Timestamp'], errors='coerce')
    else:
        print("Timestamp column not found.")

    # 2. Replace 'Infinity' or invalid strings in numeric columns
    print("Cleaning invalid numeric values (e.g., 'Infinity', 'NaN' strings)...")
    for col in df_all.columns:
        if df_all[col].dtype == object:
            # skip columns that must remain text
            if col in ["Flow ID", "Src IP", "Dst IP", "Attack Name", "source_file"]:
                continue
            # try converting
            df_all[col] = pd.to_numeric(df_all[col], errors='ignore')

    # 3. Ensure all numeric columns are float64
    numeric_cols = df_all.select_dtypes(include=['int64', 'float64']).columns
    print("Converting all numeric columns to float64 for consistency...")
    df_all[numeric_cols] = df_all[numeric_cols].astype('float64')

    # 4. Ensure Label is int64
    if "Label" in df_all.columns:
        df_all["Label"] = df_all["Label"].astype("int64")

    print("Data type standardization completed!")
    print(df_all.dtypes)

    # Export cleaned dataset
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_all.to_csv(output_path, index=False)
    print(f"Cleaned dataset exported to {output_path}")

if __name__ == "__main__":
    clean_data("../data/silver/data_merge_and_cleaned.csv")
