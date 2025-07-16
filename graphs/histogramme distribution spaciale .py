import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
from pathlib import Path
from scipy.stats import chi2_contingency


class PolarHistogramByRank:
    """
    • Filtre les RFID12 dont le suffixe (3 derniers chiffres) correspond
      aux animaux de *rank* choisi (1, 2 ou 3) dans mice_archetypes_all_data.csv
    • Trace un histogramme du % d'animaux dans chaque zone :
        – 5 s avant l'appui (Pré‑press)
        – 1 s après l'appui (Post‑press)
        – baseline aléatoire (Random)
    """

    def __init__(self, db_csv_paths, arche_csv, target_coords,
                 rank_value="1", post_delay_ms=5000, pre_delay_ms=5000,
                 random_n=10_000):

        self.db_csv_paths = db_csv_paths
        self.arche_csv = Path(arche_csv)
        self.target = target_coords  # conservé si besoin futur
        self.rank_value = str(rank_value)       # "1" | "2" | "3"
        self.post_delay_ms = post_delay_ms      # +5 s
        self.pre_delay_ms = pre_delay_ms        # −5 s
        self.random_n = random_n

        # zones (x1,y1) coin HG ; (x2,y2) coin BD
        self.zones = {
            "A": ((90, 60), (253, 162)),
            "B": ((259, 60), (420, 162)),
            "C": ((90, 260), (420, 360))
        }
        self.zlist = "ABC"

        # suffixes du rank choisi
        self.rank_suffixes = self._load_rank_suffixes()

        # dataframe concaténée + RFID filtrés
        self.df, self.rfids = self._load_and_filter()

        # print RFIDs retenus
        print(f"\n=== RFID rank {self.rank_value} représentés ===")
        for r in self.rfids:
            print(f"{r}  (suffixe {r[-3:]})")
        print("========================================\n")

        # containers
        self.pre_pos, self.post_pos, self.rand_pos = [], [], []

    # ---------- helpers ----------
    def _load_rank_suffixes(self):
        df = pd.read_csv(self.arche_csv, dtype=str)
        if self.rank_value == "male":
            ranks = {"1", "3"}
            filt = df["rank"].isin(ranks)
        else:
            filt = df["rank"] == self.rank_value
        return set(
            df[filt]["ID_Animal"].str.replace(r"\D", "", regex=True).str[-3:]
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

    # ---------- zone utils ----------
    def _zone_of(self, x, y):
        for z, ((x1, y1), (x2, y2)) in self.zones.items():
            if x1 <= x <= x2 and y1 <= y <= y2:
                return z
        return None

    # ---------- positions (pré / post) ----------
    def _collect_positions(self, time_offset_ms, pos_list):
        """Collecte les zones des animaux *rank* à un décalage temporel donné."""
        press = self.df[self.df['LEVER_PRESS'].isin(self.rfids)]

        for t0, r_press in zip(press['TIMESTAMP'].astype(float), press['LEVER_PRESS']):
            row = self.df[self.df['TIMESTAMP'] == t0 + time_offset_ms]
            if row.empty:
                continue
            row = row.iloc[0]
            for r in (x for x in self.rfids if x != r_press):
                mx, my = row.get(f"MASS_X_{r}"), row.get(f"MASS_Y_{r}")
                if any(pd.isna([mx, my])) or min(mx, my) < 0:
                    continue
                z = self._zone_of(mx, my)
                if z:
                    pos_list.append(z)

    def compute_positions(self):
        """Remplit self.pre_pos et self.post_pos."""
        self._collect_positions(-self.pre_delay_ms, self.pre_pos)
        self._collect_positions(self.post_delay_ms, self.post_pos)

    # ---------- baseline aléatoire ----------
    def compute_random(self):
        rows = self.df.sample(n=self.random_n, random_state=42)
        for _, row in rows.iterrows():
            for r in self.all_rfids:
                mx, my = row.get(f"MASS_X_{r}"), row.get(f"MASS_Y_{r}")
                if any(pd.isna([mx, my])) or min(mx, my) < 0:
                    continue
                z = self._zone_of(mx, my)
                if z:
                    self.rand_pos.append(z)

    # ---------- histogramme ----------
    def plot_histogram(self, title):
        zones = list(self.zlist)
        rand = [self.rand_pos.count(z) for z in zones]
        post = [self.post_pos.count(z) for z in zones]
        pre = [self.pre_pos.count(z) for z in zones]

        rand_tot, post_tot, pre_tot = sum(rand), sum(post), sum(pre)
        if not (post_tot and rand_tot and pre_tot):
            print("Pas assez de données pour histogramme.")
            return

        p_rand = np.array(rand) / rand_tot
        p_post = np.array(post) / post_tot
        p_pre = np.array(pre) / pre_tot

        se_rand = np.sqrt(p_rand * (1 - p_rand) / rand_tot) * 100
        se_post = np.sqrt(p_post * (1 - p_post) / post_tot) * 100
        se_pre = np.sqrt(p_pre * (1 - p_pre) / pre_tot) * 100

        rand_pct, post_pct, pre_pct = p_rand * 100, p_post * 100, p_pre * 100

        x = np.arange(len(zones))
        w = 0.25
        fig, ax = plt.subplots()
        bars = []

        bars += ax.bar(x - w, pre_pct, w, yerr=se_pre, capsize=4, color='orange', alpha=.6, label='Pré‑press (−5 s)')
        bars += ax.bar(x, post_pct, w, yerr=se_post, capsize=4, color='orange', alpha=.7, label='Post‑press (+5 s)')
        bars += ax.bar(x + w, rand_pct, w, yerr=se_rand, capsize=4, color='grey', alpha=.5, label='Random')


        ax.set_xticks(x)
        ax.set_xticklabels(zones)
        ax.set_ylabel('% observations')
        ax.set_ylim(0, 60)
        ax.set_title(f'Distribution spatiale par zone – Storer (±SE)')
        ax.legend()

        # ------- Tests statistiques (chi²) -------
        comparisons = [
            ('Pre‑press', pre, 'Post‑press', post, -w),
            ('Pre‑press', pre, 'Random', rand, w),
            ('Post‑press', post, 'Random', rand, 0),
        ]

        for name1, data1, name2, data2, offset in comparisons:
            chi2, p, _, _ = chi2_contingency([data1, data2])
            if p < 0.05:
                max_height = max(np.array(data1) + np.array(data2)) * 100 / max(pre_tot, post_tot, rand_tot)
                ax.text(x[1] + offset, max_height + 3, '*', ha='center', va='bottom', fontsize=16)

        plt.tight_layout()
        out_dir = Path(r"C:\Users\I9_1\Desktop\données csv pour python\figure clement")
        base_name = title.replace(" ", "_").replace("–", "-").lower()

        # Sauvegarde .png et .eps
        png_path = out_dir / f"{base_name}.png"
        eps_path = out_dir / f"{base_name}.eps"

        fig.savefig(png_path, dpi=300)
        fig.savefig(eps_path, format='eps')

        print(f"Figure sauvegardée :\n- {png_path}\n- {eps_path}")
        plt.show()

# -------------------------- MAIN ----------------------------
if __name__ == "__main__":
    LEVER, FEEDER = (250, 350), (265, 65)

    choice = input("Cible (levier/feeder) : ").strip().lower()
    target = LEVER if choice == "levier" else FEEDER
    title = "Direction levier" if choice == "levier" else "Direction feeder"

    rank_in = input("Quel rank afficher ? (1/2/3/male) : ").strip().lower()
    if rank_in not in {"1", "2", "3", "male"}:
        raise ValueError("Rank doit être 1, 2, 3 ou 'male'.")

    csv_files = glob.glob(r"C:\\Users\\I9_1\\Desktop\\données csv pour python\\clement dataframeM2\\DB*.csv")
    if not csv_files:
        raise FileNotFoundError("Aucun DB_*.csv trouvé")

    plotter = PolarHistogramByRank(
        db_csv_paths=csv_files,
        arche_csv=r"C:\\Users\\I9_1\\Desktop\\données csv pour python\\mice_archetypes_all_data.csv",
        target_coords=target,
        rank_value=rank_in
    )

    plotter.compute_positions()
    plotter.compute_random()
    plotter.plot_histogram(f"{title} – rank {rank_in}")