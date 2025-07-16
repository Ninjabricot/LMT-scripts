
import pandas as pd
import sqlite3
import numpy as np
import os
import re
from concurrent.futures import ProcessPoolExecutor

class MouseDataProcessor:
    def __init__(
        self,
        db_path,
        event_csv_path=None,
        output_dir=r"C:\Users\I9_1\Desktop\LMT\dataframeM2",
        output_csv_path=None,
        date_str=None,
    ):
        self.db_path = db_path
        self.output_dir = output_dir

        if date_str is None:
            m = re.search(r"(\d{8})", db_path)
            date_str = m.group(1) if m else "unknown_date"
        self.date_str = date_str

        if output_csv_path is None:
            m_num = re.search(r"females(\d{2})_", db_path, re.I)
            num = m_num.group(1) if m_num else "unknown"
            output_csv_path = os.path.join(
                output_dir, f"DB_{num}_{date_str}.csv"
            )
        self.output_csv_path = output_csv_path

        if event_csv_path is None:
            event_csv_path = self.detect_event_csv()
        self.event_csv_path = event_csv_path

        self.conn = None
        self.df = None
        self.agg_df = None
        self.final = None
        self.rfid_list = None

    def detect_event_csv(self):
        m = re.search(r"females(\d{2})_", os.path.basename(self.db_path), re.I)
        num2 = m.group(1) if m else None
        if not num2:
            print(f"⚠️ Pas trouvé de numéro foodXX dans {self.db_path}")
            return None

        target = f"event_EFAU0{num2}_{self.date_str}.csv"
        candidate = os.path.join(self.output_dir, target)
        if os.path.isfile(candidate):
            print(f"📂 Fichier événement détecté automatiquement : {candidate}")
            return candidate
        print(f"⚠️ Fichier événement non trouvé : {candidate}")
        return None

    def connect_db(self):
        self.conn = sqlite3.connect(self.db_path)

    def load_data(self):
        query = """
        SELECT 
            D.FRAMENUMBER,
            D.ANIMALID,
            D.MASS_X,
            D.MASS_Y,
            D.FRONT_X,
            D.FRONT_Y,
            D.BACK_X,
            D.BACK_Y,
            F.TIMESTAMP
        FROM DETECTION D
        JOIN FRAME F ON D.FRAMENUMBER = F.FRAMENUMBER
        """
        self.df = pd.read_sql_query(query, self.conn)

    def preprocess(self):
        self.df['TIMESTAMP'] = pd.to_datetime(self.df['TIMESTAMP'], unit='ms')
        ts_ns = self.df['TIMESTAMP'].astype(np.int64)
        bin_ns = 200_000_000
        self.df['TIME_BIN'] = pd.to_datetime((ts_ns // bin_ns) * bin_ns)
        self.df['DX'] = self.df['FRONT_X'] - self.df['BACK_X']
        self.df['DY'] = self.df['FRONT_Y'] - self.df['BACK_Y']
        self.df['DIRECTION'] = np.arctan2(self.df['DY'], self.df['DX'])

    def aggregate(self):
        self.agg_df = self.df.groupby(['TIME_BIN', 'ANIMALID']).agg({
            'MASS_X': 'mean',
            'MASS_Y': 'mean',
            'FRONT_X': 'mean',
            'FRONT_Y': 'mean',
            'DIRECTION': 'mean'
        }).reset_index()

        self.agg_df['FORMATTED_TIME'] = self.agg_df['TIME_BIN'].dt.strftime('%m/%d  %H:%M:%S:%f').str[:-3]
        self.agg_df['TIMESTAMP'] = self.agg_df['TIME_BIN'].astype(np.int64) // 10**6

    def pivot_and_format(self):
        position_cols = ['MASS_X', 'MASS_Y', 'FRONT_X', 'FRONT_Y', 'DIRECTION']
        pivot = self.agg_df.pivot_table(index=['FORMATTED_TIME', 'TIMESTAMP'], columns='ANIMALID', values=position_cols)
        pivot.columns = [f'{metric}_{int(float(animalid))}' for metric, animalid in pivot.columns]
        pivot = pivot.reset_index()

        ordered_cols = ['FORMATTED_TIME', 'TIMESTAMP']
        animal_ids = sorted(set(int(col.split('_')[-1]) for col in pivot.columns if col not in ordered_cols))
        for aid in animal_ids:
            for metric in ['MASS_X', 'MASS_Y', 'FRONT_X', 'FRONT_Y', 'DIRECTION']:
                col = f'{metric}_{aid}'
                if col in pivot.columns:
                    ordered_cols.append(col)
        self.final = pivot[ordered_cols]

    def replace_animalid_with_rfid(self):
        animal_map = pd.read_sql_query("SELECT ID, RFID FROM ANIMAL", self.conn)
        animal_map['ID'] = animal_map['ID'].astype(int).astype(str)
        animal_map['RFID'] = animal_map['RFID'].astype(str)
        id_to_rfid = dict(zip(animal_map['ID'], animal_map['RFID']))
        self.rfid_list = list(id_to_rfid.values())

        new_columns = []
        for col in self.final.columns:
            if '_' in col and col not in ['FORMATTED_TIME', 'TIMESTAMP']:
                base, aid = col.rsplit('_', 1)
                try:
                    aid_str = str(int(float(aid)))
                except ValueError:
                    aid_str = aid
                rfid = id_to_rfid.get(aid_str, aid_str)
                new_columns.append(f"{base}_{rfid}")
            else:
                new_columns.append(col)
        self.final.columns = new_columns

    def merge_lever_press_with_rfid(self):
        if not self.event_csv_path:
            self.final["LEVER_PRESS"] = "000000000000"
            return

        events = pd.read_csv(
            self.event_csv_path,
            sep=';',
            header=None,
            names=['event_type', 'event_target', 'event_time', 'rfid'],
            dtype={'rfid': str}
        )

        events['event_time'] = pd.to_datetime(events['event_time'], format='%d-%m-%Y %H:%M:%S:%f', errors='coerce')
        events.dropna(subset=['event_time'], inplace=True)

        lever_presses = events[
            (events['event_type'] == 'id_lever') & (events['rfid'].notna())
        ].copy()

        lever_presses['rfid'] = lever_presses['rfid'].str.zfill(12)
        lever_presses['FORMATTED_TIME'] = lever_presses['event_time'].dt.strftime('%m/%d  %H:%M:%S:%f').str[:-3]

        lever_dict = dict(zip(lever_presses['FORMATTED_TIME'], lever_presses['rfid']))
        self.final['LEVER_PRESS'] = self.final['FORMATTED_TIME'].map(lever_dict).fillna("000000000000")

    def export_csv(self):
        if self.output_csv_path:
            self.final.to_csv(self.output_csv_path, index=False)

    def run(self):
        self.connect_db()
        self.load_data()
        self.preprocess()
        self.aggregate()
        self.pivot_and_format()
        self.replace_animalid_with_rfid()
        self.merge_lever_press_with_rfid()
        self.export_csv()
        return self.output_csv_path  # pour suivi éventuel

def process_db(db_path):
    processor = MouseDataProcessor(db_path)
    processor.run()
    print(f"Terminé : {db_path}")

if __name__ == "__main__":
    db_paths = [
        r"C:\Users\I9_1\Desktop\données csv pour python\db_male\Expe1_Single_lever_food_females36_20220311.sqlite",
        r"C:\Users\I9_1\Desktop\données csv pour python\db_male\Expe1_Single_lever_food_females37_20220311.sqlite",
        r"C:\Users\I9_1\Desktop\données csv pour python\db_male\Expe1_Single_lever_food_females38_20220408.sqlite",
        r"C:\Users\I9_1\Desktop\données csv pour python\db_male\Expe1_Single_lever_food_females39_20220408.sqlite",
        r"C:\Users\I9_1\Desktop\données csv pour python\db_male\Expe1_Single_lever_food_females40_20220415.sqlite",
        r"C:\Users\I9_1\Desktop\données csv pour python\db_male\Expe1_Single_lever_food_females46_20220430.sqlite",
        r"C:\Users\I9_1\Desktop\données csv pour python\db_male\Expe1_Single_lever_food_females48_20220506.sqlite",
        r"C:\Users\I9_1\Desktop\données csv pour python\db_male\Expe1_Single_lever_food_females49_20220506.sqlite",
        r"C:\Users\I9_1\Desktop\données csv pour python\db_male\Expe1_Single_lever_food_females50_20220513.sqlite",
        r"C:\Users\I9_1\Desktop\données csv pour python\db_male\Expe1_Single_lever_food_females51_20220513.sqlite",
        r"C:\Users\I9_1\Desktop\données csv pour python\db_male\Expe1_Single_lever_food_females56_20220703.sqlite",
        r"C:\Users\I9_1\Desktop\données csv pour python\db_male\Expe1_Single_lever_food_females57_20220703.sqlite",
        r"C:\Users\I9_1\Desktop\données csv pour python\db_male\Expe1_Single_lever_food_females58_20220709.sqlite",
        r"C:\Users\I9_1\Desktop\données csv pour python\db_male\Expe1_Single_lever_food_females59_20220709.sqlite"
        # ▶️  ajoute autant de bases que tu veux
    ]

    max_workers = 24

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        executor.map(process_db, db_paths)
