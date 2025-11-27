# %%
# import mlx.core as mx
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("./data/loan_approval.csv")
df = df.drop("customer_id", axis=1)

# %%
print("DF Head")
print(df.head())
print("DF Info")
print(df.info())
print("DF Describe")
print(df.describe())
print("DF Shape")
print(df.shape)


'''Univariate Analysis'''
# %%
# start with the categorical features
categorical_features = df.select_dtypes("object").columns.to_list()
categorical_feature_labels = []

for cf in categorical_features:
    categorical_feature_labels.append(df[cf].unique())
categorical_feature_labels

# %%
# Plot histograms for each categorical feature
fig, axes = plt.subplots(len(categorical_features), 1, figsize=(10, 4*len(categorical_features)))

# Handle case where there's only one categorical feature
if len(categorical_features) == 1:
    axes = [axes]

for idx, cf in enumerate(categorical_features):
    df[cf].value_counts().plot(kind='bar', ax=axes[idx])
    axes[idx].set_title(f'Distribution of {cf}')
    axes[idx].set_xlabel(cf)
    axes[idx].set_ylabel('Count')
    axes[idx].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

# %%
# Plot histograms for numerical features
numerical_features = df.select_dtypes(include=[np.number]).columns.to_list()

fig, axes = plt.subplots(len(numerical_features), 1, figsize=(10, 4*len(numerical_features)))

# Handle case where there's only one numerical feature
if len(numerical_features) == 1:
    axes = [axes]

for idx, nf in enumerate(numerical_features):
    axes[idx].hist(df[nf].dropna(), bins=30, edgecolor='black')
    axes[idx].set_title(f'Distribution of {nf}')
    axes[idx].set_xlabel(nf)
    axes[idx].set_ylabel('Frequency')

plt.tight_layout()
plt.show()

'''Bivariate Analysis'''
# %%
# Box plots for numerical features comparing approved vs not approved
fig, axes = plt.subplots(len(numerical_features), 1, figsize=(10, 4*len(numerical_features)))

# Handle case where there's only one numerical feature
if len(numerical_features) == 1:
    axes = [axes]

for idx, nf in enumerate(numerical_features):
    df.boxplot(column=nf, by='loan_status', ax=axes[idx])
    axes[idx].set_title(f'Box Plot of {nf} by Loan Status')
    axes[idx].set_xlabel('Loan Status')
    axes[idx].set_ylabel(nf)

plt.tight_layout()
plt.show()