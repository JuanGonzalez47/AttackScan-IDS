import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.model_selection import train_test_split
import pickle


def train_rf_gold(input_path, model_path, reports_dir):
	os.makedirs(reports_dir, exist_ok=True)
	os.makedirs(os.path.dirname(model_path), exist_ok=True)
	os.makedirs(os.path.dirname(input_path), exist_ok=True)
	df = pd.read_csv(input_path)
	y = df["Attack_Encoded"]
	X = df.drop(columns=["Attack Name", "Attack_Encoded", "Label", "Dst IP"], errors="ignore")
	X_train, X_test, y_train, y_test = train_test_split(
		X, y, test_size=0.2, random_state=42, stratify=y
	)
	rf = RandomForestClassifier(
		n_estimators=200,
		min_samples_split=10,
		min_samples_leaf=1,
		max_features=None,
		max_depth=50,
		bootstrap=True,
		random_state=42,
		n_jobs=-1,
		class_weight="balanced_subsample"
	)
	rf.fit(X_train, y_train)
	y_pred = rf.predict(X_test)
	acc = accuracy_score(y_test, y_pred)
	macro_f1 = f1_score(y_test, y_pred, average="macro")
	weighted_f1 = f1_score(y_test, y_pred, average="weighted")
	attack_names = df[["Attack Name", "Attack_Encoded"]].drop_duplicates().sort_values("Attack_Encoded")
	target_names = attack_names["Attack Name"].tolist()
	report = classification_report(y_test, y_pred, target_names=target_names)

	with open(os.path.join(reports_dir, "rf_metrics.txt"), "w", encoding="utf-8") as f:
		f.write(f"Accuracy: {acc}\nMacro F1: {macro_f1}\nWeighted F1: {weighted_f1}\n\nClassification Report:\n{report}\n")

	feature_names = X_train.columns
	importances = rf.feature_importances_
	importance_df = pd.DataFrame({"Feature": feature_names, "Importance": importances}).sort_values(by="Importance", ascending=False)
	plt.figure(figsize=(10, 12))
	sns.barplot(data=importance_df.head(20), x="Importance", y="Feature", orient="h", palette="viridis")
	plt.title("Top 20 Most Important Features – Random Forest")
	plt.xlabel("Feature Importance")
	plt.ylabel("Feature")
	plt.tight_layout()
	plt.savefig(os.path.join(reports_dir, "rf_feature_importances.png"))
	plt.close()

	with open(model_path, "wb") as f:
		pickle.dump(rf, f)
	print(f"Modelo y reportes guardados en {reports_dir}")

if __name__ == "__main__":
	train_rf_gold(
		input_path="../data/gold/dataset_gold.csv",
		model_path="../models/random_forest_best.pkl",
		reports_dir="../reports"
	)
