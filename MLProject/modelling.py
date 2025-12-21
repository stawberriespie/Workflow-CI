import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 1. Load dataset hasil preprocessing
df = pd.read_csv("penguins_preprocessing.csv")

X = df.drop(columns=["species"])
y = df["species"]

# 2. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Aktifkan MLflow autolog
mlflow.set_experiment("Penguins_Classification")
mlflow.sklearn.autolog(log_models=True, log_input_examples=False)

# 4. Training model
with mlflow.start_run():
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # 5. Evaluasi
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print("Accuracy:", acc)