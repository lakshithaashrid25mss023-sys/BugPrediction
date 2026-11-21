import pandas as pd
from scipy.io import arff
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

# 1. Load the JM1 dataset
data, _ = arff.loadarff("data/raw/jm1.arff")
df = pd.DataFrame(data)

# 2. Convert b'Y' and b'N' into normal text
df["label"] = df["label"].str.decode("utf-8")

# 3. Choose only five simple code metrics
features = [
    "LOC_TOTAL",
    "CYCLOMATIC_COMPLEXITY",
    "BRANCH_COUNT",
    "NUM_OPERATORS",
    "NUM_OPERANDS"
]

X = df[features]

# Y means defective module = 1; N means non-defective module = 0
y = df["label"].map({"N": 0, "Y": 1})

# 4. Keep 80% for training and 20% for testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# 5. Train the Random Forest model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced"
)

model.fit(X_train, y_train)

# 6. Test the model
predictions = model.predict(X_test)

print("Accuracy:", round(accuracy_score(y_test, predictions), 4))
print("Precision:", round(precision_score(y_test, predictions), 4))
print("Recall:", round(recall_score(y_test, predictions), 4))
print("F1 Score:", round(f1_score(y_test, predictions), 4))

# 7. Save both the model and the feature order
joblib.dump(
    {
        "model": model,
        "features": features
    },
    "models/bug_model.pkl"
)

print("\nModel saved successfully in models/bug_model.pkl")