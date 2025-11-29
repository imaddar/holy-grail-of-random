# %% Imports
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from decision_tree import DecisionTree
# from decision_tree import DecisionTree
# %% One-hot encode categorical features
df = pd.read_csv("./data/loan_approval.csv")
df = df.drop("customer_id", axis=1)

categorical_features = df.select_dtypes("object").columns.to_list()
numerical_features = df.select_dtypes(include=[np.number]).columns.to_list()

encoder = OneHotEncoder(sparse_output=False)
encoded_features = encoder.fit_transform(df[categorical_features])
encoded_df = pd.DataFrame(
    encoded_features,
    columns=encoder.get_feature_names_out(categorical_features),
    index=df.index
)

df = df.drop(categorical_features, axis=1)
df = pd.concat([df, encoded_df], axis=1)

# %% Scaling
scaler = StandardScaler()
scaled = scaler.fit_transform(df[numerical_features])
df[numerical_features] = scaled

# Build X/y as NumPy arrays
X = df.drop("loan_status", axis=1).to_numpy()
y = df["loan_status"].to_numpy()
print(X.shape)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)


# %%
# Generate correlation matrix
# correlation_matrix = df.corr()
# plt.figure(figsize=(12, 10))
# sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', center=0)
# plt.title('Correlation Matrix')
# plt.tight_layout()
# plt.show()
# %% here goes the decision tree logic
tree = DecisionTree()
tree.fit(X_train, y_train)

preds = tree.predict(X_test)

accuracy = (preds == y_test).mean()
print("Accuracy:", accuracy)
