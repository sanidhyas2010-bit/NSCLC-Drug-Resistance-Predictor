import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import warnings

warnings.filterwarnings('ignore') 

print("Booting up the Data Wrangler...")

# 1. Load the "Rosetta Stone" Mapping File
print("Loading OmicsProfiles.csv to map DNA profiles to Cell Lines...")
try:
    profiles = pd.read_csv('OmicsProfiles.csv')
    # Create a dictionary that translates 'PR-...' to 'ACH-...'
    profile_to_model = dict(zip(profiles['ProfileID'], profiles['ModelID']))
except FileNotFoundError:
    print("ERROR: Could not find 'OmicsProfiles.csv'. Please download it from DepMap!")
    exit()

# 2. Load Model (Cell Line) Metadata
print("Loading Model.csv to identify Lung Cancer cells...")
models = pd.read_csv('Model.csv')
models = models.loc[:, ~models.columns.duplicated()]
cancer_col = 'OncotreeLineage' if 'OncotreeLineage' in models.columns else 'OncotreePrimaryDisease'
lung_cancer_models = set(models[models[cancer_col].astype(str).str.contains('Lung', na=False)]['ModelID'])

# Helper function to find exact gene columns
def get_gene_col(df, gene):
    for col in df.columns:
        if col.startswith(f"{gene} (") or col == gene:
            return col
    return [col for col in df.columns if gene in col][0]

# Helper function to automatically detect and translate PR- to ACH-
def standardize_ids(df, mapping_dict):
    col = df.columns[0]
    sample_val = str(df[col].iloc[0])
    
    # If already ACH-, just rename the column so Pandas recognizes it
    if sample_val.startswith('ACH-'):
        df = df.rename(columns={col: 'ModelID'})
    # If PR-, rename the column and mathematically translate the IDs to ACH-
    elif sample_val.startswith('PR-'):
        df = df.rename(columns={col: 'ProfileID'})
        df['ModelID'] = df['ProfileID'].map(mapping_dict)
    return df

# 3. Load the Hotspot Mutations (The Gas Pedal)
print("Loading Hotspot mutations for the EGFR 'Gas Pedal'...")
hotspots = pd.read_csv('OmicsSomaticMutationsMatrixHotspot.csv')
hotspots = hotspots.loc[:, ~hotspots.columns.duplicated()]
hotspots = standardize_ids(hotspots, profile_to_model)

egfr_hotspot_col = get_gene_col(hotspots, 'EGFR')
tp53_hotspot_col = get_gene_col(hotspots, 'TP53')

hotspot_egfr_cells = set(hotspots[hotspots[egfr_hotspot_col] > 0]['ModelID'])
hotspot_tp53_cells = set(hotspots[hotspots[tp53_hotspot_col] > 0]['ModelID'])

# 4. Load the Damaging Mutations (The Brakes)
print("Loading Damaging mutations for the TP53 'Brakes' (This is a massive file, please wait)...")
damaging = pd.read_csv('OmicsSomaticMutationsMatrixDamaging.csv')
damaging = damaging.loc[:, ~damaging.columns.duplicated()]
damaging = standardize_ids(damaging, profile_to_model)

tp53_damaging_col = get_gene_col(damaging, 'TP53')
damaging_tp53_cells = set(damaging[damaging[tp53_damaging_col] > 0]['ModelID'])

# 5. Find the "Super-Cancer" Cells
print("Cross-referencing biological profiles...")
broken_tp53_cells = hotspot_tp53_cells.union(damaging_tp53_cells)
super_cancer_lines = list(lung_cancer_models.intersection(hotspot_egfr_cells).intersection(broken_tp53_cells))
print(f"Isolated {len(super_cancer_lines)} extremely aggressive 'Super-Cancer' cell lines.")

if len(super_cancer_lines) == 0:
    print("Error: Still finding 0 cell lines. Check your CSV column formatting.")
    exit()

# 6. Load the PRISM Drug Sensitivity Data
print("Loading PRISM drug screening data...")
dr_file = glob.glob('*secondary-screen-dose-response-curve-parameters.csv')[0]
info_file = glob.glob('*secondary-screen-replicate-collapsed-treatment-info.csv')[0]

dr_data = pd.read_csv(dr_file)
treatment_info = pd.read_csv(info_file)

if 'depmap_id' in dr_data.columns:
    dr_data = dr_data.rename(columns={'depmap_id': 'ModelID'})

dr_data = dr_data.loc[:, ~dr_data.columns.duplicated()]
dr_super = dr_data[dr_data['ModelID'].isin(super_cancer_lines)].copy()

# 7. Map the messy drug IDs to real English drug names
print("Translating chemical IDs to English drug names...")
drug_dict = dict(zip(treatment_info['broad_id'], treatment_info['name']))
dr_super['Drug_Name'] = dr_super['broad_id'].map(drug_dict)

# 8. Calculate Drug Efficacy (AUC)
print("Calculating mathematically best backup treatments...")
avg_auc = dr_super.groupby('Drug_Name')['auc'].mean().sort_values(ascending=True)
top_10 = avg_auc.head(10)

print("\n==========================================")
print("  REAL PRISM DATA: TOP 10 BACKUP DRUGS  ")
print("==========================================")
print(top_10)
print("==========================================\n")

# 9. Draw the Chart
print("Drawing professional bar chart...")
plt.figure(figsize=(12, 7))
sns.barplot(x=top_10.values, y=top_10.index, palette='magma')

plt.title('Top 10 Most Effective Drugs Against EGFR+TP53 "Super-Cancer"\n(Real DepMap/PRISM Clinical Data)', fontsize=15, fontweight='bold')
plt.xlabel('Average AUC (Lower Score = Better at Killing Cancer)', fontsize=12)
plt.ylabel('Drug Name', fontsize=12)
plt.tight_layout()

plt.savefig('phase2_real_data_chart.png')
print("SUCCESS: Chart saved as 'phase2_real_data_chart.png' in your VS Code folder!")