import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor

print("Loading data and training the AI...")
df = pd.read_csv("master_research_data.csv")

# 1. Prepare the math grid
X = pd.crosstab(df['SANGER_MODEL_ID'], df['gene_symbol']).clip(upper=1)
y = df.groupby('SANGER_MODEL_ID')['LN_IC50'].mean()

# 2. Train the Random Forest
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# 3. Extract the top 10 genes
importances = pd.Series(model.feature_importances_, index=X.columns)
top_10 = importances.sort_values(ascending=False).head(10)

print("Generating professional bar chart...")
# 4. Set up the visual style
plt.figure(figsize=(10, 6))
sns.set_theme(style="whitegrid")

# 5. Draw the bar chart (using the 'viridis' color palette for a scientific look)
sns.barplot(x=top_10.values, y=top_10.index, hue=top_10.index, palette="viridis", legend=False)

# 6. Add professional labels and titles
plt.title("Top Genetic Predictors of Erlotinib Resistance in NSCLC", fontsize=16, fontweight='bold', pad=15)
plt.xlabel("Predictive Importance (Random Forest)", fontsize=12)
plt.ylabel("Mutated Gene", fontsize=12)

# 7. Save the high-resolution image to your folder
plt.tight_layout()
plt.savefig("erlotinib_resistance_chart.png", dpi=300)
print("Success! Chart saved as 'erlotinib_resistance_chart.png'.")