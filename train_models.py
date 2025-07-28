import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
import pickle
import os

# Make sure the models folder exists
os.makedirs("models", exist_ok=True)

# === Churn Model ===
df_churn = pd.read_csv("data/sample_churn.csv")
X_churn = df_churn[["Tenure", "MonthlyCharges"]]
y_churn = [0, 1, 0]  # Dummy labels

model_churn = LogisticRegression()
model_churn.fit(X_churn, y_churn)

with open("models/churn_model.pkl", "wb") as f:
    pickle.dump(model_churn, f)
print("✅ churn_model.pkl saved.")

# === Cluster Model ===
df_cluster = pd.read_csv("data/sample_customers.csv")
model_cluster = KMeans(n_clusters=3, random_state=42)
model_cluster.fit(df_cluster)

with open("models/cluster_model.pkl", "wb") as f:
    pickle.dump(model_cluster, f)
print("✅ cluster_model.pkl saved.")
