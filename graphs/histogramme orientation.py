import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
from pathlib import Path


# -------------------------  CLASS  --------------------------
class PolarHistogramByRank:
    def __init__(self, db_csv_paths, arche_csv, target_coords,
                 rank_value="1", delay_ms=1000, random_n=10_000):

        self.db_csv_paths = db_csv_paths
        self.arche_csv = Path(arche_csv)
        self.target = target_coords
        self.rank_value = str(rank_value)
        self.delay_ms = delay_ms
        self.random_n = random_n

        self.zones = {
            "A": ((90, 60), (253, 162)),
            "B": ((259, 60), (420, 162)),
            "C": ((90, 260), (420, 360))
        }
        self.zlist = "ABC"

        self.rank_suffixes = self._load_rank_suffixes()
        self.df, self.rfids = self._load_and_filter()

        print(f"\n=== RFID rank {self.rank_value} représentés ===")
        for r in self.rfids:
            print(f"{r}  (suffixe {r[-3:]})")
        print("========================================\n")

        self.ang = {z: [] for z in self.zlist}
        self.rand = {z: [] for z in self.zlist}
        self.post_pos, self.rand_pos = [], []

    def _load_rank_suffixes(self):
        df = pd.read_csv(self.arche_csv, dtype=str)
        return set(
            df[df["rank"] == self.rank_value]["ID_Animal"]
            .str.replace(r"\D", "", regex=True)
            .str[-3:]
        )

    def _load_and_filter(self):
        dfs = [pd.read_csv(p, dtype={'LEVER_PRESS': str}) for p in self.db_csv_paths]
        df = pd.concat(dfs, ignore_index=True)
        df['TIMESTAMP'] = pd.to_numeric(df['TIMESTAMP'], errors='coerce')

        all_rfids = [c.split('_')[-1] for c in df.columns if c.startswith("MASS_X_")]
        self.all_rfids = all_rfids
        rfids = [r for r in all_rfids if r[-3:] in self.rank_suffixes]
        if not rfids:
            raise ValueError(f"Aucun RFID rank {self.rank_value} trouvé.")
        return df, rfids

    def _zone_of(self, x, y):
        for z, ((x1, y1), (x2, y2)) in self.zones.items():
            if x1 <= x <= x2 and y1 <= y <= y2:
                return z
        return None

    def compute_angles(self):
        tx, ty = self.target
        press = self.df[self.df['LEVER_PRESS'].isin(self.rfids)]

        for t0, r_press in zip(press['TIMESTAMP'].astype(float), press['LEVER_PRESS']):
            row = self.df[self.df['TIMESTAMP'] == t0 + self.delay_ms]
            if row.empty:
                continue
            row = row.iloc[0]

            for r in (x for x in self.rfids if x != r_press):
                mx, my = row.get(f"MASS_X_{r}"), row.get(f"MASS_Y_{r}")
                fx, fy = row.get(f"FRONT_X_{r}"), row.get(f"FRONT_Y_{r}")
                if any(pd.isna([mx, my, fx, fy])) or min(mx, my, fx, fy) < 0:
                    continue
                z = self._zone_of(mx, my)
                if not z:
                    continue
                v1, v2 = np.array([fx - mx, fy - my]), np.array([tx - mx, ty - my])
                if not (np.linalg.norm(v1) and np.linalg.norm(v2)):
                    continue
                ang = (np.arctan2(*v2[::-1]) - np.arctan2(*v1[::-1])) % (2 * np.pi)
                self.ang[z].append(ang)
                self.post_pos.append(z)

    def compute_random(self):
        tx, ty = self.target
        rows = self.df.sample(n=self.random_n, random_state=42)

        for _, row in rows.iterrows():
            for r in self.all_rfids:
                mx, my = row.get(f"MASS_X_{r}"), row.get(f"MASS_Y_{r}")
                fx, fy = row.get(f"FRONT_X_{r}"), row.get(f"FRONT_Y_{r}")
                if any(pd.isna([mx, my, fx, fy])) or min(mx, my, fx, fy) < 0:
                    continue
                z = self._zone_of(mx, my)
                if not z:
                    continue
                v1, v2 = np.array([fx - mx, fy - my]), np.array([tx - mx, ty - my])
                if not (np.linalg.norm(v1) and np.linalg.norm(v2)):
                    continue
                ang = (np.arctan2(*v2[::-1]) - np.arctan2(*v1[::-1])) % (2 * np.pi)
                self.rand[z].append(ang)
                self.rand_pos.append(z)

    def plot_polar(self, title):
        bins = np.linspace(0, 2 * np.pi, 13)
        centers = (bins[:-1] + bins[1:]) / 2
        labels = {'A': 'Zone A‑water', 'B': 'Zone B‑feeder', 'C': 'Zone C‑lever'}

        fig, axs = plt.subplots(1, 3, subplot_kw={'projection': 'polar'}, figsize=(15, 5))
        for ax, z in zip(axs, self.zlist):
            if self.ang[z]:
                h, _ = np.histogram(self.ang[z], bins)
                ax.bar(centers, h / h.sum(), width=np.diff(bins),
                       color='orange', alpha=.7, edgecolor='orange', label='Post‑press (+1s)')
            if self.rand[z]:
                h, _ = np.histogram(self.rand[z], bins)
                ax.bar(centers, h / h.sum(), width=np.diff(bins),
                       color='grey', alpha=.6, edgecolor='grey', label='Random')
            ax.set_title(labels[z])
            ax.set_theta_zero_location('N')
            ax.set_theta_direction(-1)
            ax.set_xticks(np.deg2rad(np.arange(0, 360, 45)))
            ax.set_ylim(0, 0.3)
            ax.legend(frameon=False, fontsize=7)

        fig.suptitle(title, fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        save_dir = r"C:\Users\I9_1\Desktop\données csv pour python\figure clement"
        fname = f"{save_dir}/polar_{self.rank_value}_{title.replace(' ', '_')}"
        fig.savefig(fname + ".eps", format='eps')
        fig.savefig(fname + ".png", format='png')
        plt.show()

    def plot_histogram(self, title):
        zones = list(self.zlist)
        rand = [self.rand_pos.count(z) for z in zones]
        post = [self.post_pos.count(z) for z in zones]
        post_tot, rand_tot = sum(post), sum(rand)
        if not (post_tot and rand_tot):
            print("Pas assez de données pour histogramme.")
            return

        p_post, p_rand = np.array(post) / post_tot, np.array(rand) / rand_tot
        se_post = np.sqrt(p_post * (1 - p_post) / post_tot) * 100
        se_rand = np.sqrt(p_rand * (1 - p_rand) / rand_tot) * 100
        post_pct, rand_pct = p_post * 100, p_rand * 100

        x = np.arange(len(zones))
        w = .35
        fig, ax = plt.subplots()
        ax.bar(x - w / 2, post_pct, w, yerr=se_post, capsize=4,
               color='orange', alpha=.7, label='Post‑press')
        ax.bar(x + w / 2, rand_pct, w, yerr=se_rand, capsize=4,
               color='grey', alpha=.5, label='Random')
        ax.set_xticks(x)
        ax.set_xticklabels(zones)
        ax.set_ylabel('% observations')
        ax.set_ylim(0, 60)
        ax.set_title('spacial distribution Storer (±SE)')
        ax.legend()
        plt.tight_layout()

        save_dir = r"C:\Users\I9_1\Desktop\données csv pour python\figure clement"
        fname = f"{save_dir}/histogram_{self.rank_value}_{title.replace(' ', '_')}"
        fig.savefig(fname + ".eps", format='eps')
        fig.savefig(fname + ".png", format='png')
        plt.show()


# -------------------------- MAIN ----------------------------
if __name__ == "__main__":
    LEVER, FEEDER = (250, 350), (265, 65)

    choice = input("Cible (levier/feeder) : ").strip().lower()
    target = LEVER if choice == "levier" else FEEDER
    title = "Direction levier" if choice == "levier" else "Direction feeder"

    rank_in = input("Quel rank afficher ? (1/2/3) : ").strip()
    if rank_in not in {"1", "2", "3"}:
        raise ValueError("Rank doit être 1, 2 ou 3.")

    csv_files = glob.glob(r"C:\Users\I9_1\Desktop\données csv pour python\clement dataframeM2\DB*.csv")
    if not csv_files:
        raise FileNotFoundError("Aucun DB_*.csv trouvé")

    plotter = PolarHistogramByRank(
        db_csv_paths=csv_files,
        arche_csv=r"C:\Users\I9_1\Desktop\données csv pour python\mice_archetypes_all_data.csv",
        target_coords=target,
        rank_value=rank_in
    )
    plotter.compute_angles()
    plotter.compute_random()
    plotter.plot_polar(f"{title} – Storer")
    plotter.plot_histogram(f"{title} – rank {rank_in}")
