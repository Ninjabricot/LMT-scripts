import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


df = pd.read_csv(r"C:\Users\I9_1\Desktop\LMT\dataframeM2\DB_03_31032021.csv")
df['TIMESTAMP'] = pd.to_numeric(df['TIMESTAMP'], errors='coerce')
rfid_ids = sorted(set(col.split('_')[-1] for col in df.columns if 'DIRECTION_' in col))

colors = ['red', 'green', 'blue']
rfid_colors = dict(zip(rfid_ids, colors))

fig, axs = plt.subplots(1, 3, figsize=(18, 6))
titles = ['2 sec avant appui', 'Moment de l\'appui', '2 sec après appui']
time_offsets = [-2000, 0, 2000]

# Lignes avec appui levier
df_lever = df[df['LEVER_PRESS'] != 0]
total_presses = len(df_lever)

# Pour chaque offset temporel, tracer les flèches
for ax, title, offset in zip(axs, titles, time_offsets):
    ax.set_title(title)
    ax.set_xlabel('MASS_X')
    ax.set_ylabel('MASS_Y')

    for idx, lever_row in df_lever.iterrows():
        target_time = lever_row['TIMESTAMP'] + offset

        # Convertir LEVER_PRESS en RFID formaté (12 chiffres avec zéros en tête)
        try:
            lever_rfid = f"{int(float(lever_row['LEVER_PRESS'])):012d}"
        except (ValueError, TypeError):
            continue  # Ignore cette ligne si la valeur n'est pas convertible

        # Trouver la ligne la plus proche dans le temps
        closest_idx = (df['TIMESTAMP'] - target_time).abs().idxmin()
        row = df.loc[closest_idx]

        for rfid in rfid_ids:
            if rfid == lever_rfid:
                continue  # Ne pas tracer l’animal qui a appuyé

            x_col = f'MASS_X_{rfid}'
            y_col = f'MASS_Y_{rfid}'
            dir_col = f'DIRECTION_{rfid}'

            if pd.notna(row[x_col]) and pd.notna(row[y_col]) and pd.notna(row[dir_col]):
                x = row[x_col]
                y = row[y_col]
                direction = row[dir_col]
                arrow_length = 10
                dx = arrow_length * np.cos(direction)
                dy = arrow_length * np.sin(direction)

                ax.arrow(x, y, dx, dy, head_width=2, head_length=2,
                         fc=rfid_colors[rfid], ec=rfid_colors[rfid])

    ax.grid(True)
    ax.axis('equal')

handles = [plt.Line2D([], [], color=rfid_colors[rfid], lw=3, label=f'RFID {rfid}') for rfid in rfid_ids]
handles.append(plt.Line2D([], [], color='black', lw=0, label=f'Total appuis levier : {total_presses}'))
axs[2].legend(handles=handles, loc='upper right')

plt.tight_layout()
plt.show()
