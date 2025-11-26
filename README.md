# AttackScan-IDS

Intrusion Detection System for IoT Networks using Machine Learning

---

## Description
AttackScan-IDS is a complete pipeline for automatic detection of attacks in IoT network traffic. It includes data cleaning, exploratory analysis, preprocessing, modeling, and interactive visualization in a Streamlit dashboard.

## Project Structure

```
├── data/
│   ├── bronze/         # Raw data
│   ├── silver/         # Cleaned and merged data
│   └── gold/           # Final preprocessed data
├── models/             # Trained models and encoders
├── Notebooks/          # Jupyter notebooks for analysis and development
├── reports/
│   └── figures/        # Plots and results
├── src/
│   ├── dashboard/      # Streamlit dashboard
│   ├── evaluate.py     # Model evaluation
│   ├── predict.py      # Offline prediction
│   ├── preprocessing.py# Data cleaning and transformation
│   ├── train.py        # Model training
│   └── utils/          # Helper functions
├── requirements.txt    # Dependencies
└── README.md           # This file
```

## Installation

1. Clone the repository:
   ```
   git clone <repo-url>
   cd AttackScan-IDS
   ```
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. (Optional) Create and activate a virtual environment:
   ```
   python -m venv .venv
   .venv\Scripts\activate
   ```

## Dashboard Usage

1. Run the dashboard:
   ```
   streamlit run src/dashboard/dashboard.py
   ```
2. Explore the tabs:
   - **Introduction**: Project summary and objectives.
   - **EDA**: Interactive dataset visualization and conclusions.
   - **Model Results**: Metrics, classification report, and feature importances.
   - **Online Prediction**: Paste a JSON to predict the attack type.

## Data Pipeline

- **Bronze**: Raw original data.
- **Silver**: Cleaned and merged data (`data_merge_and_cleaned.csv`).
- **Gold**: Final preprocessed and scaled data (`dataset_gold.csv`).

## Main Scripts

- `preprocessing.py`: Data cleaning and transformation.
- `train.py`: Random Forest model training.
- `evaluate.py`: Model evaluation and metrics report.
- `predict.py`: Offline prediction for new data.
- `dashboard.py`: Interactive Streamlit dashboard.

## Models and Encoders

- Models and encoders are stored in the `models/` folder:
  - `random_forest_best.pkl`: Trained model.
  - `scaler.pkl`: StandardScaler for feature scaling.
  - `label_encoder_attack_name.pkl`: Attack name label encoder.
  - `label_encoder_dst_ip.pkl`: Destination IP label encoder.

## Online Prediction Example

In the prediction tab, paste a JSON like:
```json
{
  "Src Port": 52344,
  "Dst IP": "192.168.137.250",
  "Dst Port": 80,
  "Flow Duration": 120000,
  "Total Fwd Packet": 8,
  "Total Bwd packets": 10,
  ...
}
```
The system will preprocess and display the predicted attack type.

## Recommendations
- For better control and traceability of the pipeline, run the notebooks in the `Notebooks/` folder sequentially (from 01 to 06), so you can visualize and validate each step before moving forward.
- If you prefer automation, run the scripts in order: cleaning → preprocessing → training → evaluation.
- Keep models and encoders in the `models/` folder up to date.
- For new data, follow the pipeline and use the dashboard for visualization and prediction.

## Credits
Developed by JuanGonzalez47 and collaborators.

---

Questions or suggestions? Open an issue in the repository.
