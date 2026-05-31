from pathlib import Path
import re
import pandas as pd
import numpy as np

""""Extract training Data from training_trajectories folder"""

T_cols = ["Date","Time","Lon","Lat","Depth","Mag","MagType","Source"]
def read_Y_files(path):
    path = Path(path)
    df = pd.read_csv(path)
    df["T"] = [1,2,3]
    return df

T_cols = ["ID","Date" ,"Time","Lon","Lat","Depth","Mag","MagType","Source"]
def read_T_files(path):
    path = Path(path)
    vals = pd.read_csv(path, header=None).iloc[:,0].tolist()
    df = pd.DataFrame([vals], columns=T_cols)
    return df

def load_raw_data(train_data_path, min_mag = 3.0, max_obs = None, seed=123):
    train_data_path = Path(train_data_path)
    X_files = list(train_data_path.glob("X_*.csv"))
    ids = []
    for path in X_files:
        match = re.match(r"X_(\d+)\.csv$", path.name)
        if match is not None: ids.append(int(match.group(1)))
    ids = sorted(ids)
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    print(f"Using shuffled IDs with seed = {seed}")
    
    df_X_list, df_Y_list, valid_ids = [], [], []
    skipped_ids = []

    for n in ids:
        X_path = train_data_path / f"X_{n}.csv"
        T1_path = train_data_path / f"T1_{n}.csv"
        T2_path = train_data_path / f"T2_{n}.csv"
        T3_path = train_data_path / f"T3_{n}.csv"

        required_paths = [X_path, T1_path, T2_path, T3_path]
        if not all(p.exists() for p in required_paths):
            missing = [p.name for p in required_paths if not p.exists()]
            skipped_ids.append(n)
            continue

        df_X = pd.read_csv(X_path)
        # Filter out magnitudes < 3
        if min_mag is not None:
            df_X["Mag"] = pd.to_numeric(df_X["Mag"], errors="coerce")
            df_X = df_X[df_X["Mag"] >= min_mag].reset_index(drop=True)
        if len(df_X) == 0: skipped_ids.append(n); continue

        df_T1 = read_T_files(T1_path)
        df_T2 = read_T_files(T2_path)
        df_T3 = read_T_files(T3_path)
        df_Y = pd.concat([df_T1, df_T2, df_T3], axis =0, ignore_index=True)
        df_Y["T"] = [1,2,3]
        df_X_list.append(df_X)
        df_Y_list.append(df_Y)
        valid_ids.append(n)

        if max_obs is not None and len(valid_ids) >= max_obs:
            break
    print("# of skipped ids", len(skipped_ids))
    print("# of valid T", len(valid_ids))
    return df_X_list, df_Y_list, valid_ids
