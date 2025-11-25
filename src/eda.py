# eda.py
# Exploratory Data Analysis pipeline extracted from 03_exploratory_analysis.ipynb

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import kstest, f_oneway, kruskal

def run_eda(input_path):
    """
    Load cleaned dataset, show class distribution, feature statistics, correlations, and save figures.
    """
    df = pd.read_csv(input_path)
    import os
    figures_dir = 'reports/figures'
    os.makedirs(figures_dir, exist_ok=True)
    # 1. Class distribution (Benign vs Attack)
    plt.figure(figsize=(6,4))
    df['Label'].value_counts().plot(kind='bar')
    plt.title("Class Distribution (Benign vs Attack)")
    plt.xlabel("Class (0 = Benign, 1 = Attack)")
    plt.ylabel("Count")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'class_distribution.png'))
    plt.close()
    print("Class distribution:")
    print(df['Label'].value_counts())

    # 2. Feature statistics
    print("Basic Statistical Summary:")
    print(df.describe().T)

    # 3. Histogram and boxplot per feature
    def plot_feature_distribution(df, col):
        plt.figure(figsize=(16, 6))
        plt.subplot(1, 2, 1)
        sns.histplot(df[col], bins=50, kde=True)
        plt.title(f"Histogram - {col}")
        plt.subplot(1, 2, 2)
        sns.boxplot(x=df[col])
        plt.title(f"Boxplot - {col}")
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, f'{col}_hist_box.png'))
        plt.close()

    # Example: plot for first 5 numeric features
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    for col in numeric_cols[:5]:
        plot_feature_distribution(df, col)

    # 4. Pairplot por grupo de características
    def pairplot_group(df, features, sample_size=5000, title="Pairplot"):
        valid_features = [col for col in features if col in df.columns]
        if len(valid_features) < 2:
            print(f"Not enough valid features for: {title}")
            return
        df_sample = df[valid_features + ["Label"]].sample(min(sample_size, len(df)), random_state=42)
        sns.pairplot(df_sample, vars=valid_features, hue="Label", diag_kind="kde", plot_kws={"alpha": 0.5, "s": 10})
        plt.suptitle(title, fontsize=18)
        plt.savefig(os.path.join(figures_dir, f'{title}_pairplot.png'))
        plt.close()

    # Example groups (from notebook)
    flow_features = ["Flow Duration", "Flow Bytes/s", "Flow Packets/s", "Fwd Packets/s", "Bwd Packets/s", "Down/Up Ratio"]
    pairplot_group(df, flow_features, title="Flow Features")

    # 5. Correlation heatmap
    plt.figure(figsize=(18,14))
    sns.heatmap(df[numeric_cols].corr(), cmap="coolwarm", center=0, vmin=-1, vmax=1)
    plt.title("Correlation Heatmap – Numeric Features")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'correlation_heatmap.png'))
    plt.close()

    # 6. Statistical tests
    
    classes = df["Attack Name"].unique().tolist() if "Attack Name" in df.columns else df["Label"].unique().tolist()
    normality_results = {}
    stats_results = {}
    def ks_normality_test(series):
        series = series.dropna()
        if series.std() == 0:
            return 1.0
        series_std = (series - series.mean()) / series.std()
        stat, p = kstest(series_std, "norm")
        return p
    def anova_test(df, col):
        groups = [df[df["Attack Name"] == cl][col].dropna() for cl in classes]
        stat, p = f_oneway(*groups)
        return p
    def kruskal_test(df, col):
        groups = [df[df["Attack Name"] == cl][col].dropna() for cl in classes]
        stat, p = kruskal(*groups)
        return p
    print("Running normality and class-comparison tests...")
    for col in numeric_cols:
        p_norm = ks_normality_test(df[col])
        is_normal = p_norm > 0.05
        normality_results[col] = {"p_normality": p_norm, "normal": is_normal}
        if is_normal:
            p_stat = anova_test(df, col)
            test_used = "ANOVA"
        else:
            p_stat = kruskal_test(df, col)
            test_used = "Kruskal–Wallis"
        stats_results[col] = {"test_used": test_used, "p_value": p_stat, "significant": p_stat < 0.05}
    print("Tests completed.")

    # 7. Temporal analysis of Timestamp
    if "Timestamp" in df.columns:
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
        df["hour"] = df["Timestamp"].dt.hour
        plt.figure(figsize=(10,5))
        df.groupby(['hour', 'Label']).size().unstack(fill_value=0).plot(kind='bar', stacked=True)
        plt.title("Records per Hour: Benign vs Attack")
        plt.xlabel("Hour of Day")
        plt.ylabel("Count")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, 'records_per_hour.png'))
        plt.close()
        plt.figure(figsize=(16,6))
        df.groupby(['hour', 'Attack Name']).size().unstack(fill_value=0).plot(kind='bar', stacked=True, colormap='tab20')
        plt.title("Traffic per Hour by Attack Type")
        plt.xlabel("Hour of Day")
        plt.ylabel("Count")
        plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1))
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, 'traffic_per_hour_by_attack.png'))
        plt.close()
        df['date'] = pd.to_datetime(df['Timestamp']).dt.date
        traffic_per_day = df.groupby('date').size()
        plt.figure(figsize=(14,5))
        traffic_per_day.plot(kind='line', marker='o')
        plt.title("Total Traffic per Day")
        plt.xlabel("Date")
        plt.ylabel("Number of Records")
        plt.grid(alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, 'traffic_per_day.png'))
        plt.close()

    print("EDA completed. Figures and stats saved.")

if __name__ == "__main__":
    run_eda("../data/silver/data_merge_and_cleaned.csv")
