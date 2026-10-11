import pandas as pd
import numpy as np
import gseapy as gp
import matplotlib.pyplot as plt

print("1. Loading pure NSCLC data...")
df = pd.read_csv('phase3_RNA_survival_mechanisms_NSCLC.csv')
df.rename(columns={df.columns[0]: 'Gene_Info'}, inplace=True)
df['Gene_Symbol'] = df['Gene_Info'].apply(lambda x: str(x).split(' ')[0])

print("2. Calculating the 'Signed Statistic' (ChatGPT Recommendation)...")
# We multiply the direction (sign of Fold Change) by the mathematical weight (-log10 p-value)
df['sign'] = np.sign(df['Log2_Fold_Change'])
df['signed_stat'] = df['sign'] * -np.log10(df['p_value'])

# Clean and sort the ranked list
ranking_df = df[['Gene_Symbol', 'signed_stat']].dropna().sort_values('signed_stat', ascending=False)
ranking_df = ranking_df.drop_duplicates(subset=['Gene_Symbol'])

print("3. Executing Ranked GSEA against 19,219 genes...")
# This scans the entire genome distribution to find pathway-level significance
pre_res = gp.prerank(rnk=ranking_df,
                     gene_sets='GO_Biological_Process_2021',
                     threads=4,
                     min_size=5,
                     max_size=1000,
                     permutation_num=1000,
                     outdir=None,
                     seed=42)

results_df = pre_res.res2d

# Filter for pathways enriched in the Resistant cells (Positive NES) with a standard GSEA FDR < 0.25
significant_pathways = results_df[(results_df['FDR q-val'] < 0.25) & (results_df['NES'] > 0)].sort_values('NES', ascending=False).head(10)

if significant_pathways.empty:
    print("\nNo statistically significant pathways found. The sample size of 61 is heavily limiting.")
else:
    print("\n=== TOP VALIDATED BIOLOGICAL PATHWAYS (RANKED GSEA) ===")
    
    plt.style.use('dark_background')
    plt.figure(figsize=(10, 6))
    
    significant_pathways['Term'] = significant_pathways['Term'].apply(lambda x: (x[:45] + '...') if len(x) > 45 else x)
    
    # Plotting by Normalized Enrichment Score (NES)
    plt.barh(significant_pathways['Term'][::-1], significant_pathways['NES'][::-1], color='mediumorchid')
    
    plt.title('Baseline Pathways in Resistant Lung Cancer (Ranked GSEA)')
    plt.xlabel('Normalized Enrichment Score (NES)')
    plt.tight_layout()
    plt.savefig('phase3_gsea_chart.png')
    
    for index, row in significant_pathways.iterrows():
        print(f"- {row['Term']} (FDR: {row['FDR q-val']:.3f}, NES: {row['NES']:.2f})")

    plt.show()