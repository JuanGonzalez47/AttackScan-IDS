# preproc.py
# Preprocessing pipeline extracted from 04_preprocessing.ipynb

import pandas as pd
import os
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder

def preprocess(input_path, output_path):
    """
    Load cleaned dataset, scale features, encode labels, select features, and export GOLD dataset.
    """
    # Crear directorio de salida si no existe
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Cargar datos
    df = pd.read_csv(input_path)

    # 1. Eliminar columnas irrelevantes
    irrelevant_features = [
        "Protocol", "Bwd PSH Flags", "Bwd URG Flags", "Fwd Bytes/Bulk Avg", "Fwd Packet/Bulk Avg", "Fwd Bulk Rate Avg",
        "PSH Flag Count", "FIN Flag Count", "ECE Flag Count", "CWR Flag Count"
    ]
    cols_to_drop = [col for col in irrelevant_features if col in df.columns]
    df.drop(columns=cols_to_drop, inplace=True)

    # 2. Eliminar multicolineales
    redundant_features = [
        "Bwd Packet Length Max", "Bwd Packet Length Std", "Bwd Segment Size Avg", "Total Length of Bwd Packet", "Bwd IAT Total", "Bwd IAT Max", "Bwd IAT Std",
        "Fwd Packet Length Max", "Fwd Packet Length Std", "Fwd Segment Size Avg", "Total Length of Fwd Packet", "Fwd IAT Total", "Fwd IAT Max", "Fwd IAT Std",
        "Packet Length Max", "Packet Length Std", "Packet Length Variance", "Average Packet Size",
        "Idle Max", "Idle Min", "Idle Std", "Active Max", "Active Min", "Active Std",
        "Fwd Act Data Pkts", "Fwd Header Length", "Bwd Header Length", "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min",
        "Flow Packets/s", "Bwd Packets/s", "Fwd Packets/s", "Flow Bytes/s", "Bwd IAT Min", "Fwd IAT Min"
    ]
    cols_to_drop = [c for c in redundant_features if c in df.columns]
    df.drop(columns=cols_to_drop, inplace=True)

    # 3. Eliminar identificadores y columnas de ruido
    id_columns = ["Flow ID", "Timestamp", "source_file", "Src IP"]
    cols_to_drop = [col for col in id_columns if col in df.columns]
    df.drop(columns=cols_to_drop, inplace=True)

    # 4. Estandarización de variables numéricas
    numeric_features = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    numeric_features = [c for c in numeric_features if c not in ["Label", "Attack Name"]]
    scaler = StandardScaler()
    df_scaled = df.copy()
    df_scaled[numeric_features] = scaler.fit_transform(df[numeric_features])

    # 5. Codificación de Attack Name y Dst IP
    le = LabelEncoder()
    le_ip = LabelEncoder()
    if "Attack Name" in df_scaled.columns:
        df_scaled["Attack_Encoded"] = le.fit_transform(df_scaled["Attack Name"])
    if "Dst IP" in df_scaled.columns:
        df_scaled["DstIP_Encoded"] = le_ip.fit_transform(df_scaled["Dst IP"])

    # 6. Exportar GOLD dataset
    df_scaled.to_csv(output_path, index=False)
    print(f"GOLD dataset exported to {output_path}")

    # 7. Exportar los encoders
    models_dir = "../models"
    os.makedirs(models_dir, exist_ok=True)
    with open(os.path.join(models_dir, "label_encoder_attack_name.pkl"), "wb") as f:
        pickle.dump(le, f)
    print(f"LabelEncoder for attack_name exported to: {os.path.join(models_dir, 'label_encoder_attack_name.pkl')}")
    if "Dst IP" in df_scaled.columns:
        with open(os.path.join(models_dir, "label_encoder_dst_ip.pkl"), "wb") as f:
            pickle.dump(le_ip, f)
        print(f"LabelEncoder for dst_ip exported to: {os.path.join(models_dir, 'label_encoder_dst_ip.pkl')}")

if __name__ == "__main__":
    preprocess("../data/silver/data_merge_and_cleaned.csv", "../data/gold/dataset_gold.csv")
