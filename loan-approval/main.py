# %% Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from decision_tree import DecisionTree

# %%
# ---------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------

def evaluate_model(model, X, y, name="set"):
    preds = model.predict(X)
    acc = (preds == y).mean()
    print(f"Accuracy on {name}: {acc:.4f}")
    return preds

def show_confusion_matrix(y_true, y_pred, labels=None, title="Confusion Matrix"):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    print(title)
    print(cm)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(title)
    plt.tight_layout()
    plt.show()

    return cm

# %%
# ---------------------------------------------------------------------
# Load & Preprocess Data
# ---------------------------------------------------------------------

df = pd.read_csv("./data/loan_approval.csv")
df = df.drop("customer_id", axis=1)

categorical_features = df.select_dtypes("object").columns.to_list()
numerical_features = df.select_dtypes(include=[np.number]).columns.to_list()

# --- One-hot encode categorical variables ---
encoder = OneHotEncoder(sparse_output=False)
encoded = encoder.fit_transform(df[categorical_features])
encoded_df = pd.DataFrame(
    encoded,
    columns=encoder.get_feature_names_out(categorical_features),
    index=df.index
)

df = df.drop(categorical_features, axis=1)
df = pd.concat([df, encoded_df], axis=1)

# --- Scale numerical features ---
scaler = StandardScaler()
scaled = scaler.fit_transform(df[numerical_features])
df[numerical_features] = scaled

# --- Convert to NumPy arrays ---
X = df.drop("loan_status", axis=1).to_numpy()
y = df["loan_status"].to_numpy()

print("Data shape:", X.shape)

# %%
# ---------------------------------------------------------------------
# Train / Val / Test Split
# ---------------------------------------------------------------------

X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full,
    test_size=0.25,      # 0.25 * 0.8 = 20%
    random_state=42
)

print("Train:", X_train.shape)
print("Val:  ", X_val.shape)
print("Test: ", X_test.shape)

# %%
# ---------------------------------------------------------------------
# Train Decision Tree
# ---------------------------------------------------------------------

tree = DecisionTree(
    min_samples_split=2000,
    max_depth=12
)

tree.fit(X_train, y_train)

# %%
# ---------------------------------------------------------------------
# Evaluate on Train, Val, Test
# ---------------------------------------------------------------------

pred_train = evaluate_model(tree, X_train, y_train, "TRAIN")
pred_val   = evaluate_model(tree, X_val, y_val, "VALIDATION")
pred_test  = evaluate_model(tree, X_test, y_test, "TEST")

# %%
# ---------------------------------------------------------------------
# Confusion Matrices
# ---------------------------------------------------------------------

# show_confusion_matrix(y_train, pred_train, labels=[0, 1], title="Train Confusion Matrix")
# show_confusion_matrix(y_val,   pred_val,   labels=[0, 1], title="Validation Confusion Matrix")
# show_confusion_matrix(y_test,  pred_test,  labels=[0, 1], title="Test Confusion Matrix")
