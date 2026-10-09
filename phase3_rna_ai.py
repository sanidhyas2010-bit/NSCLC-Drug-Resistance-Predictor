import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt

print("1. Loading PRISM drug data...")
df_prism = pd.read_csv('prism-repurposing-20q2-secondary-screen-dose-response-curve-parameters.csv', low_memory=False)
df_clean = df_prism.dropna(subset=['name', 'depmap_id', 'auc']).copy()
drug_data = df_clean[df_clean['name'].str.lower() == 'sangivamycin']

median_auc = drug_data['auc'].median()
resistant_cells = drug_data[drug_data['auc'] > median_auc]['depmap_id'].tolist()
sensitive_cells = drug_data[drug_data['auc'] <= median_auc]['depmap_id'].tolist()

print("\n2. Loading the 395MB RNA dataset...")
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

print(f"\n3. MATCH SUCCESS: Found {len(available_resistant)} Resistant and {len(available_sensitive)} Sensitive cells!")

res_rna = rna_data.loc[available_resistant].select_dtypes(include=['number'])
sen_rna = rna_data.loc[available_sensitive].select_dtypes(include=['number'])

# 4. Differential Gene Expression
log2_fold_change = res_rna.mean() - sen_rna.mean()
t_stats, p_values = stats.ttest_ind(res_rna, sen_rna, equal_var=False, nan_policy='omit')

dge_results = pd.DataFrame({'Log2_Fold_Change': log2_fold_change, 'p_value': p_values}, index=res_rna.columns).dropna()

# 5. Apply FDR Penalty
reject, fdr_pvalues, _, _ = multipletests(dge_results['p_value'], alpha=0.05, method='fdr_bh')
dge_results['FDR_Adjusted_p_value'] = fdr_pvalues
dge_results = dge_results.sort_values(by='p_value')
dge_results.to_csv('phase3_RNA_survival_mechanisms.csv')

# 6. Generate the Volcano Plot
print("\n4. Generating Volcano Plot...")
plt.style.use('dark_background')
plt.figure(figsize=(10, 6))

# Fix the squished graph by zooming in on the real data
plt.xlim(-2, 2)

plt.scatter(dge_results['Log2_Fold_Change'], -np.log10(dge_results['p_value']), color='dimgray', alpha=0.5, label='Static/Noise')

# LOWERED THRESHOLD: Catching genes that shifted by 0.5 instead of 1.0
upregulated = dge_results[(dge_results['Log2_Fold_Change'] > 0.5) & (dge_results['p_value'] < 0.05)]
plt.scatter(upregulated['Log2_Fold_Change'], -np.log10(upregulated['p_value']), color='crimson', alpha=0.8, label='Turned UP in Survivors')

downregulated = dge_results[(dge_results['Log2_Fold_Change'] < -0.5) & (dge_results['p_value'] < 0.05)]
plt.scatter(downregulated['Log2_Fold_Change'], -np.log10(downregulated['p_value']), color='royalblue', alpha=0.8, label='Turned DOWN in Survivors')

plt.title('RNA Defense Mechanisms: Surviving Sangivamycin')
plt.xlabel('Log2 Fold Change (Difference in RNA Volume)')
plt.ylabel('-Log10 p-value (Statistical Significance)')
plt.axvline(x=0, color='white', linestyle='--', linewidth=0.8)
plt.axhline(y=-np.log10(0.05), color='white', linestyle='--', linewidth=0.8)
plt.legend()
plt.tight_layout()
plt.savefig('phase3_volcano_plot.png')

print("\n=== TOP 5 SECRET DEFENSE GENES ===")
print(upregulated[['Log2_Fold_Change', 'p_value', 'FDR_Adjusted_p_value']].head())
plt.show()