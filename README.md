# Predicting Drug Resistance in Lung Cancer (NSCLC) 🧬


## Why I Built This

I built this project to see how machine learning can be applied on a real-world biological problem. Cancer drug resistance is a huge issue in clinical oncology, and I'm interested in seeing whether Python can effectively uncover the mechanisms through which resistance develops.

## What This Is
This project involves me building a custom training set out of genomic and pharmaceutical data to determine the ability of a Random Forest Regressor to predict resistance pathways in Non-Small Cell Lung Carcinoma (NSCLC).

## What This Isn't
This isn't a finished product. This is me experimenting in Python to see how effective an ML approach can be at finding interesting biological information.

## Building the Research Set
I gathered and cleaned the following official databases in Python:
GDSC2 (Genomics of Drug Sensitivity in Cancer) from the Wellcome Sanger Institute: Containing the IC50 values (drug resistances) for each cell line
Cosmic Cell Line Project: Containing the mutation sets for each cell line

## What This Found
I found that without requiring any biological information, the AI was successfully able to find known resistance pathways! The gene EGFR was found to be the main resistance pathway, which is correct as EGFR is the target of Erlotinib (an EGFR-inhibitor). The AI additionally found secondary resistance pathways, including the tumor-suppressor genes STK11 and TP53.

## Files

`analysis.py`: My Python script which merges the two sets of data and builds the Random Forest Regressor model, finding the most relevant genes. `master_research_data.csv`: The combined set of data relating NSCLC cell lines to their genetic mutations. `erlotinib_resistance_chart.png`: A bar chart displaying the top 10 most relevant genes discovered by the AI.
