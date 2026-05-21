import pandas as pd
import numpy as np

# Carica i tuoi due CSV
df_auto = pd.read_csv('OPTICAL_FLOW/farneback/results/data/RBF_pred_1.csv')
df_manual = pd.read_csv('OPTICAL_FLOW/farneback/results/data/RBF_gt_1.csv')

# Funzione per formattare Media +/- SD
def get_stats(df, col_name):
    mean_val = df[col_name].mean()
    sd_val = df[col_name].std()
    return f"{mean_val:.2f} ± {sd_val:.2f}"

# Mappatura delle colonne che ci interessano (solo Magnitude)
regions = [
    ("Whole Macula", "Global", "magnitude total"),
    ("Inner Ring", "Average", "magnitude inner avg"),
    ("Inner Ring", "Superior (225)", "magnitude 225 inner"),
    ("Inner Ring", "Inferior (45)", "magnitude 45 inner"),
    ("Inner Ring", "East (315)", "magnitude 315 inner"),
    ("Inner Ring", "West (135)", "magnitude 135 inner"),
    ("Outer Ring", "Average", "magnitude outer avg"),
    ("Outer Ring", "Superior (225)", "magnitude 225 outer"),
    ("Outer Ring", "Inferior (45)", "magnitude 45 outer"),
    ("Outer Ring", "East (315)", "magnitude 315 outer"),
    ("Outer Ring", "West (135)", "magnitude 135 outer")
]

print("Anatomical Region | Quadrant | Automated Pipeline (μm) | Manual Ground Truth (μm)")
print("--- | --- | --- | ---")

for region, quadrant, col in regions:
    auto_stats = get_stats(df_auto, col)
    manual_stats = get_stats(df_manual, col)
    
    # Stampa la riga in formato Markdown (puoi incollarla in Word o Excel)
    print(f"{region} & {quadrant} & {auto_stats} & {manual_stats} \\\\")