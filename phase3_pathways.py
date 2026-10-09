import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gseapy as gp

# 1. Load the results we just calculated
print("Loading Phase 3 RNA Volcano results...")
df = pd.read_csv('phase3_RNA_survival_mechanisms.csv')

# Rename the first column (which holds the gene names) so we can grab it
df.rename(columns={df.columns[0]: 'Gene_Info'}, inplace=True)

# 2. Clean the Gene Names
# DepMap outputs "ABCC3 (8714)". Pathway databases only understand "ABCC3".
df['Gene_Symbol'] = df['Gene_Info'].apply(lambda x: str(x).split(' ')[0])

# 3. Grab the Red Dots (The Defense Genes)
defense_genes = df[(df['Log2_Fold_Change'] > 0.5) & (df['p_value'] < 0.05)]['Gene_Symbol'].tolist()
print(f"Found {len(defense_genes)} upregulated defense genes.")

# 4. Run the Pathway Analysis
print("Consulting global biological databases (KEGG and Gene Ontology)...")
enrichment_results = gp.enrichr(gene_list=defense_genes,
                                gene_sets=['GO_Biological_Process_2021', 'KEGG_2021_Human'],
                                organism='human',
                                outdir=None)

# 5. Extract the mathematically significant pathways
results_df = enrichment_results.results
significant_pathways = results_df[results_df['Adjusted P-value'] < 0.05].sort_values('Adjusted P-value').head(10)

# 6. Generate the Bar Chart
if significant_pathways.empty:
    print("No unified pathways found. The genes might be acting independently!")
else:
    plt.style.use('dark_background')
    plt.figure(figsize=(10, 6))

    # Shorten the pathway names so they fit nicely on the chart
    significant_pathways['Term'] = significant_pathways['Term'].apply(lambda x: (x[:45] + '...') if len(x) > 45 else x)

    # Plot the chart
    plt.barh(significant_pathways['Term'][::-1], -np.log10(significant_pathways['Adjusted P-value'][::-1]), color='mediumorchid')

    plt.title('Biological Factories Hijacked by Surviving Cancer Cells')
    plt.xlabel('-Log10 Adjusted p-value (Mathematical Certainty)')
    plt.tight_layout()
    plt.savefig('phase3_pathway_chart.png')

    print("\n=== TOP BIOLOGICAL FACTORIES HIJACKED ===")
    for index, row in significant_pathways.iterrows():
        print(f"- {row['Term']} (Overlap: {row['Overlap']})")

    plt.show()