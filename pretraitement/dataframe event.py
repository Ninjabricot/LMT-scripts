import os, re, glob, pickle, sqlite3
import pandas as pd

# ---------------- METS TES DOSSIERS ICI ---------------- #
db_dirs = [
    r"\\NAS-Kinect\home\Data Live Mouse Tracker\Clement\Data_LMT_3_mice\Expe1_Single_lever_food_EFAU003\Expe1_Single_lever_food_31032021",
    r"\\NAS-Kinect\home\Data Live Mouse Tracker\Clement\Data_LMT_3_mice\Expe1_Single_lever_food_EFAU004\Expe1_Single_lever_food_26042021",
    r"\\NAS-Kinect\home\Data Live Mouse Tracker\Clement\Data_LMT_3_mice\Expe1_Single_lever_food_EFAU005\Expe1_Single_lever_food_20210503",
    r"\\NAS-Kinect\home\Data Live Mouse Tracker\Clement\Data_LMT_3_mice\Expe1_Single_lever_food_EFAU006\Expe1_Single_lever_food_20210511",
    r"\\NAS-Kinect\home\Data Live Mouse Tracker\Clement\Data_LMT_3_mice\Expe1_Single_lever_food_EFAU007\Expe1_Single_lever_food_20210526",
    r"\\NAS-Kinect\home\Data Live Mouse Tracker\Clement\Data_LMT_3_mice\Expe1_Single_lever_food_EFAU009\Expe1_Single_lever_food_20210603",
    r"\\NAS-Kinect\home\Data Live Mouse Tracker\Clement\Data_LMT_3_mice\Expe1_Single_lever_food_EFAU013\Expe1_Single_lever_food_20210618",
    r"\\NAS-Kinect\home\Data Live Mouse Tracker\Clement\Data_LMT_3_mice\Expe1_Single_lever_food_EFAU014\Expe1_Single_lever_food_20210618",
    r"\\NAS-Kinect\home\Data Live Mouse Tracker\Clement\Data_LMT_3_mice\Expe1_Single_lever_food_EFAU015\Expe1_Single_lever_food_20210625",
    r"\\NAS-Kinect\home\Data Live Mouse Tracker\Clement\Data_LMT_3_mice\Expe1_Single_lever_food_EFAU016\Expe1_Single_lever_food_20210625",
    r"\\NAS-Kinect\home\Data Live Mouse Tracker\Clement\Data_LMT_3_mice\Expe1_Single_lever_food_EFAU017\Expe1_Single_lever_food_20210723",
    r"\\NAS-Kinect\home\Data Live Mouse Tracker\Clement\Data_LMT_3_mice\Expe1_Single_lever_food_EFAU018\Expe1_Single_lever_food_20210723",
    r"\\NAS-Kinect\home\Data Live Mouse Tracker\Clement\Data_LMT_3_mice\Expe1_Single_lever_food_EFAU019\Expe1_Single_lever_food_20210723",
    r"\\NAS-Kinect\home\Data Live Mouse Tracker\Clement\Data_LMT_3_mice\Expe1_Single_lever_food_EFAU021\Expe1_Single_lever_food_20210827",
    r"\\NAS-Kinect\home\Data Live Mouse Tracker\Clement\Data_LMT_3_mice\Expe1_Single_lever_food_EFAU022\Expe1_Single_lever_food_20210827",
    r"\\NAS-Kinect\home\Data Live Mouse Tracker\Clement\Data_LMT_3_mice\Expe1_Single_lever_food_EFAU023\Expe1_Single_lever_food_20210827"


]
out_dir = r"C:\Users\I9_1\Desktop\données csv pour python\clement dataframeM2"
ZONE = dict(x_min=215, x_max=310, y_min=320, y_max=385)
chunk_size = 900

for db_dir in db_dirs:
    # 1) .sqlite
    db_files = glob.glob(os.path.join(db_dir, "*.sqlite"))
    if not db_files:
        print(f"⚠️  Pas de .sqlite dans {db_dir}")
        continue
    db_path = db_files[0]

    # 2) ID + date
    animal_id = re.search(r"EFAU\d+", db_path).group()
    date_str  = re.search(r"\d{8}(?!.*\d)", db_path).group()

    # 3) .pkl avec la même date
    pkl_folder     = os.path.join(os.path.dirname(db_dir), "Reward_lever")
    pkls           = [p for p in glob.glob(os.path.join(pkl_folder, "*.pkl"))
                      if date_str in os.path.basename(p)]
    if not pkls:
        print(f"⚠️  Pas de .pkl {date_str} dans {pkl_folder}")
        continue
    pkl_path = pkls[0]

    with open(pkl_path, "rb") as f:
        frames_pkl = pickle.load(f)

    # 4) SQL → pandas
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA temp_store=MEMORY")
    conn.execute("PRAGMA cache_size=-1000000")
    df_anim = pd.read_sql_query("SELECT ID AS ANIMALID, RFID FROM ANIMAL", conn)

    dfs = []
    for start in range(0, len(frames_pkl), chunk_size):
        chunk = frames_pkl[start:start + chunk_size]
        q = f"""
            SELECT D.FRAMENUMBER, D.ANIMALID,
                   D.MASS_X, D.MASS_Y, F.TIMESTAMP
            FROM DETECTION D
            JOIN FRAME F ON D.FRAMENUMBER = F.FRAMENUMBER
            WHERE D.FRAMENUMBER IN ({','.join('?'*len(chunk))})
        """
        dfs.append(pd.read_sql_query(q, conn, params=chunk))
    conn.close()

    df = (pd.concat(dfs, ignore_index=True)
            .merge(df_anim, on="ANIMALID", how="left"))
    df["TIMESTAMP"] = pd.to_datetime(df["TIMESTAMP"], unit="ms")

    # 5) winner
    df['in_zone'] = (ZONE['x_min'] <= df['MASS_X']) & (df['MASS_X'] <= ZONE['x_max']) & \
                    (ZONE['y_min'] <= df['MASS_Y']) & (df['MASS_Y'] <= ZONE['y_max'])
    df_zone = (df[df['in_zone']]
               .groupby('FRAMENUMBER')
               .first()
               .reset_index())
    if df_zone.empty:
        print(f"❌  Aucune frame valide pour {animal_id} {date_str}")
        continue

    df_zone['date_fmt'] = (df_zone['TIMESTAMP']
                           .dt.floor('s')
                           .dt.strftime("%d-%m-%Y %H:%M:%S") + ":000")

    csv_df = pd.DataFrame({
        "id_lever": "id_lever",
        "lever":    "lever",
        "date":     df_zone["date_fmt"],
        "rfid":     df_zone["RFID"]
    })

    # 6) export
    csv_name = f"event_{animal_id}_{date_str}.csv"
    csv_path = os.path.join(out_dir, csv_name)
    csv_df.to_csv(csv_path, sep=";", index=False)
    print(f"✅  Export OK → {csv_path}")
