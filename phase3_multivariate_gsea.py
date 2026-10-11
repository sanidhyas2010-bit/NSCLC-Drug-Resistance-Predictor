import pandas as pd
import numpy as np
import gseapy as gp

print("=== 1. PREPARING HISTOLOGY-ADJUSTED COHORT ===")
# 1. Load mappings and metadata
profiles = pd.read_csv('OmicsProfiles.csv')
pr_to_ach = dict(zip(profiles['ProfileID'], profiles['ModelID']))

model_df = pd.read_csv('Model.csv')
nsclc_models = model_df[model_df.apply(lambda r: r.astype(str).str.contains('Non-Small Cell', case=False).any(), axis=1)]
valid_ach_ids = set(nsclc_models['ModelID'].dropna() if 'ModelID' in nsclc_models.columns else nsclc_models.iloc[:, 0].dropna())

prism = pd.read_csv('prism-repurposing-20q2-secondary-screen-dose-response-curve-parameters.csv', low_memory=False)
drug_nsclc = prism[(prism['name'].str.lower() == 'sangivamycin') & (prism['depmap_id'].isin(valid_ach_ids))]
auc_series = drug_nsclc.groupby('depmap_id')['auc'].mean()

def safe_load_matrix(filepath):
    df = pd.read_csv(filepath)
    id_col = None
    for col in df.columns[:5]:
        val = str(df[col].dropna().iloc[0])
        if 'ACH-' in val or 'PR-' in val:
            id_col = col
            break
    if id_col is not None:
        if 'PR-' in str(df[id_col].dropna().iloc[0]):
            df[id_col] = df[id_col].map(pr_to_ach)
        df = df.dropna(subset=[id_col])
        df.set_index(id_col, inplace=True)
    return df

rna = safe_load_matrix('OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv')
rna_numeric = rna.select_dtypes(include=['number'])
rna_averaged = rna_numeric.groupby(rna_numeric.index).mean()

final_ids = sorted(list(set(auc_series.index).intersection(set(rna_averaged.index))))
print(f"Total matched models (RNA + AUC): {len(final_ids)}")

# 2. Extract Histology Categories
final_models_df = model_df[model_df['ModelID'].isin(final_ids)].copy() if 'ModelID' in model_df.columns else model_df[model_df.iloc[:, 0].isin(final_ids)].copy()
id_col_model = 'ModelID' if 'ModelID' in model_df.columns else model_df.columns[0]
final_models_df.set_index(id_col_model, inplace=True)
final_models_df = final_models_df.loc[final_ids]

def categorize_histology(row):
    text = " ".join([str(val).lower() for val in row.values])
    if 'squamous' in text: return 'Squamous'
    if 'adenocarcinoma' in text: return 'Adenocarcinoma'
    return 'Other'

final_models_df['Histology'] = final_models_df.apply(categorize_histology, axis=1)
histology_counts = final_models_df['Histology'].value_counts()
print(f"Cohort Breakdown: {histology_counts.to_dict()}")

print("\n=== 2. EXECUTING VECTORIZED MULTIVARIATE OLS REGRESSION ===")
# Build Design Matrix (X) = Intercept + AUC + Histology Dummies
X = pd.DataFrame({'Intercept': 1, 'AUC': auc_series.loc[final_ids].values}, index=final_ids)
dummies = pd.get_dummies(final_models_df['Histology'], drop_first=True, dtype=float)
X = pd.concat([X, dummies], axis=1)

X_mat = X.values
Y_mat = rna_averaged.loc[final_ids].values

# OLS Vectorized Math: Beta = (X^T * X)^-1 * X^T * Y
XtX_inv = np.linalg.inv(X_mat.T @ X_mat)
beta = XtX_inv @ X_mat.T @ Y_mat

# Calculate standard errors and t-statistics
residuals = Y_mat - (X_mat @ beta)
rss = np.sum(residuals**2, axis=0)
df = X_mat.shape[0] - X_mat.shape[1]
sigma2 = rss / df

auc_idx = list(X.columns).index('AUC')
se_auc = np.sqrt(sigma2 * XtX_inv[auc_idx, auc_idx])
t_stat_auc = beta[auc_idx, :] / (se_auc + 1e-9)

ranking_df = pd.DataFrame({
    'Gene_Raw': rna_averaged.columns,
    't_stat': t_stat_auc
}).dropna()

ranking_df['Gene_Symbol'] = ranking_df['Gene_Raw'].apply(lambda s: str(s).split(' ')[0])
ranking_df = ranking_df.sort_values('t_stat', ascending=False).drop_duplicates('Gene_Symbol')

print("\n=== 3. RUNNING HISTOLOGY-ADJUSTED GSEA ===")
pre_res = gp.prerank(
    rnk=ranking_df[['Gene_Symbol', 't_stat']],
    gene_sets='GO_Biological_Process_2021',
    threads=4,
    min_size=15,
    max_size=500,
    permutation_num=1000,
    seed=42,
    outdir=None
)

res = pre_res.res2d
sig = res[(res['FDR q-val'] < 0.25) & (res['NES'] > 0)].sort_values('NES', ascending=False)

print("\nAdjusted Pathways (FDR q < 0.25):")
if sig.empty:
    print("No pathways survived histology adjustment. The original signal was entirely driven by lineage.")
else:
    for _, row in sig.head(5).iterrows():
        print(f"\nPathway: {row['Term']}")
        print(f"  NES: {row['NES']:.2f} | FDR q-val: {row['FDR q-val']:.3f}")
        leading_genes = str(row['Lead_genes']).split(';')[:8]
        print(f"  Leading-edge genes: {', '.join(leading_genes)}")