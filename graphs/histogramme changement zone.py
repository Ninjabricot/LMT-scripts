import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
from multiprocessing import Pool, cpu_count
import os


ZONES = {
    "A": ((90, 60), (253, 162)),
    "B": ((259, 60), (420, 162)),
    "C": ((90, 260), (420, 360)),  # ancienne C+D
}


def in_zone(x, y, zone):
    (x1, y1), (x2, y2) = ZONES[zone]
    return x1 <= x <= x2 and y1 <= y <= y2


def zone_of(x, y):
    for z in ZONES:
        if in_zone(x, y, z):
            return z
    return None


def process_file(path, delay_before, delay_after):
    df = pd.read_csv(path, dtype={'LEVER_PRESS': str})
    df['TIMESTAMP'] = pd.to_numeric(df['TIMESTAMP'], errors='coerce')
    df = df.dropna(subset=['TIMESTAMP']).set_index('TIMESTAMP')

    rfid_ids = sorted({c.split('_')[-1] for c in df.columns if c.startswith('MASS_X_')})
    if len(rfid_ids) != 3:
        return [], []

    transitions_1 = []
    transitions_3 = []

    press_df = df[df['LEVER_PRESS'].isin(rfid_ids)]

    for t, rfid_press in zip(press_df.index, press_df['LEVER_PRESS']):
        t_before = t + delay_before
        t_after = t + delay_after
        if t_before not in df.index or t_after not in df.index:
            continue

        row_before = df.loc[t_before]
        row_after = df.loc[t_after]
        if isinstance(row_before, pd.DataFrame):
            row_before = row_before.iloc[0]
        if isinstance(row_after, pd.DataFrame):
            row_after = row_after.iloc[0]

        others = [r for r in rfid_ids if r != rfid_press]
        for rank, rfid in zip([1, 3], others):
            xb = row_before.get(f'MASS_X_{rfid}')
            yb = row_before.get(f'MASS_Y_{rfid}')
            xa = row_after.get(f'MASS_X_{rfid}')
            ya = row_after.get(f'MASS_Y_{rfid}')
            if pd.isna([xb, yb, xa, ya]).any() or min(xb, yb, xa, ya) < 0:
                continue

            zone_before = zone_of(xb, yb)
            zone_after = zone_of(xa, ya)
            if zone_before in ['A', 'C']:
                if rank == 1:
                    transitions_1.append(zone_after == 'B')
                else:
                    transitions_3.append(zone_after == 'B')

    return transitions_1, transitions_3


def plot(transitions):
    data = {}
    ci95 = {}

    t1 = transitions[1]
    t3 = transitions[3]
    combined = t1 + t3

    def compute_pct_and_ci(lst):
        n = len(lst)
        if n == 0:
            return 0, 0
        p = sum(lst) / n
        ci = 1.96 * np.sqrt(p * (1 - p) / n) * 100
        return p * 100, ci

    data['Scrounger'], ci95['Scrounger'] = compute_pct_and_ci(t1)
    data['Worker'], ci95['Worker'] = compute_pct_and_ci(t3)
    data['Total'], ci95['Total'] = compute_pct_and_ci(combined)

    fig, ax = plt.subplots(figsize=(6, 5))
    labels = list(data.keys())
    values = [data[k] for k in labels]
    errors = [ci95[k] for k in labels]

    ax.bar(labels, values, yerr=errors, capsize=5,
           color=['blue', 'red', 'gray'], alpha=0.8)
    ax.set_ylim(0, 100)
    ax.set_ylabel('% transitions A/C → B')
    ax.set_title('Transitions à t−5s → t+3s vers zone B')

    for i, v in enumerate(values):
        ax.text(i, v + errors[i] + 1, f"{v:.1f}%", ha='center')

    plt.tight_layout()

    # Export .eps et .png
    save_dir = r"C:\Users\I9_1\Desktop\données csv pour python\figure clement"
    os.makedirs(save_dir, exist_ok=True)
    fname = os.path.join(save_dir, "transitions_AC_to_B")
    fig.savefig(fname + ".eps", format="eps")
    fig.savefig(fname + ".png", format="png")
    plt.show()


# ------------------ exécution ------------------
if __name__ == "__main__":
    delay_before = -5000
    delay_after = 3000
    csv_paths = glob.glob(r"C:\Users\I9_1\Desktop\données csv pour python\clement dataframeM2\DB*.csv")
    if not csv_paths:
        raise FileNotFoundError("Aucun CSV trouvé")

    args = [(p, delay_before, delay_after) for p in csv_paths]

    transitions = {1: [], 3: []}
    with Pool(processes=cpu_count()) as pool:
        results = pool.starmap(process_file, args)
        for t1, t3 in results:
            transitions[1].extend(t1)
            transitions[3].extend(t3)

    plot(transitions)
