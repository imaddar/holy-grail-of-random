# %% Imports
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import seaborn as sns
import numpy as np
'''Preprocessing'''
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
scaled_features = scaler.fit_transform(df[numerical_features])
scaled_df = pd.DataFrame(
    scaled_features,
    columns=numerical_features,
    index=df.index
)
df[numerical_features] = scaled_df
df.head(1)

# %%
# Generate correlation matrix
correlation_matrix = df.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', center=0)
plt.title('Correlation Matrix')
plt.tight_layout()
plt.show()
