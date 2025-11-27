# %%
# import mlx.core as mx
import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv("./data/loan_approval.csv")
# %%
print(df.columns)
# %%
# make a bar graph of approved vs not approved loans for a given numerical feature (not categorical)
feature = 'credit_history_years'
approved = df[df['loan_status'] == 1]
not_approved = df[df['loan_status'] == 0]
approved_feature = approved[feature].mean()
not_approved_feature = not_approved[feature].mean()
plt.bar(["approved", "not approved"], [approved_feature, not_approved_feature])
plt.ylabel(f'average {feature}')
plt.title(f'average {feature} of approved vs not approved loans')
plt.show()

# %%
