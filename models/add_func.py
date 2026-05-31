# Functions in the ipynb (do not need to import these to kaggle/colab to run .ipynb file)

import sys
from pathlib import Path
import importlib.util
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
import joblib
import helpers.feature_extract as fx
import helpers.load_data as ld
import helpers.evaluate as eh
import helpers.baseline_models as bsm

WINDOWS = {"T1": (0.0, 1.0), # 24 hours within
           "T2": (1.0, 3.0), # 24-72 hours
           "T3": (3.0, 7.0)} # 72 -168 hours
PLATE_MIN_COUNT = 30
# FEATURE COLS
FEATURE_COLS = ["log_time_gap_days",
                "Depth", "Mag", "Lat", "Lon",
                "dx_km", "dy_km", "log_flat_dist_km",
                "rel_time_diff", "rel_mag_diff", "rel_depth_diff"]

SCALE_COLS = ["log_time_gap_days",
                "Depth", "Mag","Lat", "Lon",
                "dx_km", "dy_km", "log_flat_dist_km",
                "rel_time_diff", "rel_mag_diff", "rel_depth_diff"]
PLATES_PATH = "models/plates/plate_mainshock_analysis/tectonicplates-master/GeoJSON/PB2002_plates"


def mainshock_ref(df_X):
    "Mainshock Reference Function"
    df_X = df_X.copy()
    df_X["datetime"] = pd.to_datetime(df_X["Date"].astype(str) + " " + df_X["Time"].astype(str),
                                      format="%Y-%m-%d %H:%M:%S")
    df_X = df_X.sort_values("datetime").reset_index(drop=True)
    idx = fx.find_mainshock_idx(df_X)
    row = df_X.loc[idx]
    return {"datetime": row["datetime"],
            "mag": float(row["Mag"])}
    
def encode_Y_one(df_X, df_Y):
    "One sequence: raw T1/T2/T3 DataFrame ==> [3, 2] (delta_mag, norm_time)."
    mainshock = mainshock_ref(df_X)
    df_Y = df_Y.copy()
    df_Y["datetime"] = pd.to_datetime(df_Y["Date"] + " " + df_Y["Time"], format="%Y-%m-%d %H:%M:%S")
    df_Y["Mag"] = pd.to_numeric(df_Y["Mag"])
    Y = np.zeros((3, 2), dtype=np.float64)

    for i, key in enumerate(("T1", "T2", "T3")):
        row = df_Y.loc[df_Y["T"] == i + 1].iloc[0]
        w0, w1 = WINDOWS[key]
        dt_days = (row["datetime"] - mainshock["datetime"]).total_seconds() / (60 * 60*24)
        Y[i, 0] = row["Mag"] - mainshock["mag"] # delta_mag
        Y[i, 1] = (dt_days - w0) / (w1 - w0) # norm_time
    return Y


def decode_predictions(pred,df_X):
    "Inverse of encode_Y_one, Model output [3,2] + raw X ===> Dataframe with predicted Time and predicted mag in abs unit"
    pred = np.asarray(pred).reshape(3, 2)
    mainshock = mainshock_ref(df_X)
    rows = []
    for i, key in enumerate(("T1", "T2", "T3")):
        w0, w1 = WINDOWS[key]
        norm_time = np.clip(float(pred[i, 1]), 0.0, 1.0)
        days = w0 + norm_time * (w1 - w0)
        predicted_time = mainshock["datetime"] + pd.to_timedelta(days, unit="D")
        rows.append({
            "T": i + 1,
            "predicted_time": predicted_time.strftime("%Y%m%d%H%M%S"),
            "predicted_mag": mainshock["mag"] + float(pred[i, 0]),
        })
    return pd.DataFrame(rows)


def build_dfX(df_X_list,valid_ids=None,plate_group_by_id=None,
              plate_vec=None):
    if valid_ids is None:
        valid_ids = list(range(len(df_X_list)))

    engineered = []

    for df, seq_id in zip(df_X_list, valid_ids):
        df_eng = fx.extract_oneX(df,plate_group = plate_group_by_id.get(seq_id, "Other"), plate_vec = plate_vec)
        engineered.append(df_eng)
    feature_cols = list(FEATURE_COLS)
    feature_cols += [f"plate_{p}" for p in plate_vec]
    return engineered, feature_cols


def build_XY_train(df_X_list, df_Y_list, valid_ids=None, scaler_path=None):
    scaler_path = Path(scaler_path)
    scaler_path.parent.mkdir(parents=True, exist_ok=True)
    if valid_ids is None:
        valid_ids = list(range(len(df_X_list)))

    plate_group_by_id, plate_vec, _, _, _ = fx.fit_plate_groups(df_X_list=df_X_list,valid_ids=valid_ids,
                                                                    plates_path=PLATES_PATH,
                                                                    min_count=PLATE_MIN_COUNT)

    engineered, feature_cols = build_dfX(df_X_list=df_X_list,
                                                      valid_ids=valid_ids,
                                                      plate_group_by_id=plate_group_by_id,
                                                      plate_vec=plate_vec)

    scaler, medians = fx.fit_scalar(engineered, SCALE_COLS)
    scaled = [fx.scale_seq(df, scaler, medians, SCALE_COLS) for df in engineered]
    X_arrays = [df[feature_cols].astype(np.float64).values for df in scaled]
    Y_arrays = [encode_Y_one(x, y) for x, y in zip(df_X_list, df_Y_list)]

    with open(scaler_path, "wb") as f:
        pickle.dump({
            "scaler": scaler,
            "medians": medians,
            "feature_cols": feature_cols,
            "scale_cols": SCALE_COLS,
            "plate_vec": plate_vec,
            "plates_path": PLATES_PATH,
            "plate_min_count": PLATE_MIN_COUNT,
        }, f)

    return X_arrays, Y_arrays, feature_cols

def build_XY_test( df_X_list, df_Y_list=None, valid_ids=None, scaler_path=None):
    scaler_path = Path(scaler_path)
    with open(scaler_path, "rb") as f:
        s = pickle.load(f)

    if valid_ids is None:
        valid_ids = list(range(len(df_X_list)))

    use_magtype = s.get("use_magtype", False)
    magtype_vec = s.get("magtype_vec", [])
    plate_vec = s.get("plate_vec", [])

    plates = fx.load_plate_polygons(s["plates_path"])
    plate_group_by_id, plate_table = fx.transform_plate_groups(df_X_list=df_X_list,
                                                                valid_ids=valid_ids,
                                                                plates=plates, train_plate_vec=plate_vec)

    engineered, _, = build_dfX(df_X_list=df_X_list,
                               valid_ids=valid_ids,
                               plate_group_by_id=plate_group_by_id,plate_vec=plate_vec)

    scaled = [fx.scale_seq(df, s["scaler"], s["medians"], s["scale_cols"]) for df in engineered]
    X_arrays = [df[s["feature_cols"]].astype(np.float64).values for df in scaled]

    if df_Y_list is None:
        return X_arrays, s["feature_cols"]
    Y_arrays = [encode_Y_one(x, y) for x, y in zip(df_X_list, df_Y_list)]
    return X_arrays, Y_arrays, s["feature_cols"]