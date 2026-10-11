import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt

print("1. Loading Model metadata to fix the 'Pan-Cancer Leak'...")
# Load Model.csv to identify only the Lung Cancer cells
model_df = pd.read_csv('Model.csv')

# Dynamically search all columns for Non-Small Cell Lung Cancer to grab the correct ACH IDs
nsclc_models = model_df[model_df.apply(lambda row: row.astype(str).str.contains('Non-Small Cell', case=False).any(), axis=1)]

# Assuming the ID column is named 'ModelID' (Standard for DepMap)
if 'ModelID' in nsclc_models.columns:
    valid_ach_ids = nsclc_models['ModelID'].tolist()
else:
    valid_ach_ids = nsclc_models.iloc[:, 0].tolist()

print(f"   Found {len(valid_ach_ids)} verified NSCLC cell lines in the database.")

print("\n2. Loading PRISM drug data and isolating NSCLC models...")
df_prism = pd.read_csv('prism-repurposing-20q2-secondary-screen-dose-response-curve-parameters.csv', low_memory=False)
df_clean = df_prism.dropna(subset=['name', 'depmap_id', 'auc']).copy()

# FILTER: Keep only Sangivamycin AND only our validated NSCLC cells
drug_data = df_clean[(df_clean['name'].str.lower() == 'sangivamycin') & (df_clean['depmap_id'].isin(valid_ach_ids))]

# Split the lung cancer cells mathematically
median_auc = drug_data['auc'].median()
resistant_cells = drug_data[drug_data['auc'] > median_auc]['depmap_id'].tolist()
sensitive_cells = drug_data[drug_data['auc'] <= median_auc]['depmap_id'].tolist()

print(f"   Filtered Cohort: {len(resistant_cells)} Resistant NSCLC cells and {len(sensitive_cells)} Sensitive NSCLC cells.")

print("\n3. Loading RNA dataset and bridging IDs...")
rna_data = pd.read_csv('OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv')

id_col = None
for col in rna_data.columns[:5]:
    sample_val = str(rna_data[col].dropna().iloc[0])
    if 'ACH-' in sample_val or 'PR-' in sample_val:
        id_col = col
        break

if 'PR-' in str(rna_data[id_col].iloc[0]):
    profiles = pd.read_csv('OmicsProfiles.csv')
    pr_to_ach = dict(zip(profiles['ProfileID'], profiles['ModelID']))
    rna_data[id_col] = rna_data[id_col].map(pr_to_ach)

rna_data.set_index(id_col, inplace=True)
available_resistant = [c for c in resistant_cells if c in rna_data.index]
available_sensitive = [c for c in sensitive_cells if c in rna_data.index]

print(f"   MATCH SUCCESS: {len(available_resistant)} Resistant and {len(available_sensitive)} Sensitive NSCLC cells matched to RNA data.")

res_rna = rna_data.loc[available_resistant].select_dtypes(include=['number'])
sen_rna = rna_data.loc[available_sensitive].select_dtypes(include=['number'])

print("\n4. Running Differential Gene Expression (DGE)...")
log2_fold_change = res_rna.mean() - sen_rna.mean()
t_stats, p_values = stats.ttest_ind(res_rna, sen_rna, equal_var=False, nan_policy='omit')

dge_results = pd.DataFrame({'Log2_Fold_Change': log2_fold_change, 'p_value': p_values}, index=res_rna.columns).dropna()

print("5. Applying FDR Penalty...")
reject, fdr_pvalues, _, _ = multipletests(dge_results['p_value'], alpha=0.05, method='fdr_bh')
dge_results['FDR_Adjusted_p_value'] = fdr_pvalues
dge_results = dge_results.sort_values(by='p_value')

# Save the pure Lung Cancer dataset
dge_results.to_csv('phase3_RNA_survival_mechanisms_NSCLC.csv')

print("\n6. Generating Volcano Plot...")
plt.style.use('dark_background')
plt.figure(figsize=(10, 6))
plt.xlim(-2, 2)

plt.scatter(dge_results['Log2_Fold_Change'], -np.log10(dge_results['p_value']), color='dimgray', alpha=0.5, label='Static/Noise')

upregulated = dge_results[(dge_results['Log2_Fold_Change'] > 0.5) & (dge_results['p_value'] < 0.05)]
plt.scatter(upregulated['Log2_Fold_Change'], -np.log10(upregulated['p_value']), color='crimson', alpha=0.8, label='High Baseline in Resistant (UP)')

downregulated = dge_results[(dge_results['Log2_Fold_Change'] < -0.5) & (dge_results['p_value'] < 0.05)]
plt.scatter(downregulated['Log2_Fold_Change'], -np.log10(downregulated['p_value']), color='royalblue', alpha=0.8, label='Low Baseline in Resistant (DOWN)')

# Updated Title to reflect ChatGPT's critique on Baseline vs Reaction
plt.title('Baseline RNA Profiles: Sangivamycin-Resistant NSCLC')
plt.xlabel('Log2 Fold Change (Difference in Baseline RNA Volume)')
plt.ylabel('-Log10 p-value (Statistical Significance)')
plt.axvline(x=0, color='white', linestyle='--', linewidth=0.8)
plt.axhline(y=-np.log10(0.05), color='white', linestyle='--', linewidth=0.8)
plt.legend()
plt.tight_layout()
plt.savefig('phase3_volcano_plot_NSCLC.png')

print("\n=== TOP 5 CANDIDATE RESISTANCE GENES (NSCLC Filtered) ===")
print(upregulated[['Log2_Fold_Change', 'p_value', 'FDR_Adjusted_p_value']].head())
plt.show()