import numpy as np
import pandas as pd
from pathlib import Path
import geopandas as gpd
from sklearn.preprocessing import StandardScaler

""""
Feature Engineering Functions (one raw X data frame => one engineered Dataframe)
Core function is extract_oneX. Everything else is a helper. 
"""

def find_mainshock_idx(seq_df):
    """
    - Find mainshock within sequence
    - Assumes the last event in X is the mainshock event
    """
    return seq_df.index[-1]



def add_flat_dist(df, main_lat, main_lon):
    """
    Approximate local Earth distance (assuming its flat) from each event to main shock
    """
    lat = df["Lat"].astype(float).values
    lon = df["Lon"].astype(float).values

    # longitude wraparound correction
    dlon_deg = (lon - main_lon + 180) % 360 -180
    dlat_deg = lat - main_lat

    main_lat_rad = np.deg2rad(main_lat)
    df["dx_km"] = 113.32 * np.cos(main_lat_rad) * dlon_deg
    df["dy_km"] =  113.3 * dlat_deg

    flat_dist = np.sqrt(df["dx_km"]**2 + df["dy_km"]**2)
    df["log_flat_dist_km"] = np.log1p(flat_dist)
    return df



def encode_time(seq_df):
    """
    - Adds Date, Time into datetime unit. 
    - Computes time gap and log time gap (t_i - t_{i-1})
    """ 
    seq_df = seq_df.copy()
    seq_df["datetime"] = pd.to_datetime(seq_df['Date'] + ' ' + seq_df['Time'],format="%Y-%m-%d %H:%M:%S")
    time_gap_sec = seq_df["datetime"].diff().dt.total_seconds().fillna(0)
    seq_df["time_gap_days"] = time_gap_sec / (60 * 60 * 24)
    seq_df["log_time_gap_days"] = np.log1p(seq_df["time_gap_days"])
    return seq_df


# Plate Divide Feature Functions
def load_plate_polygons(path: Path) -> gpd.GeoDataFrame:
    plates = gpd.read_file(path)
    plates = plates[["Code", "PlateName", "geometry"]].copy()
    plates = plates.set_crs("EPSG:4326", allow_override=True)
    return plates


def assign_plates(mainshocks: pd.DataFrame, plates: gpd.GeoDataFrame) -> pd.DataFrame:
    points = gpd.GeoDataFrame(
        mainshocks.copy(),
        geometry=gpd.points_from_xy(mainshocks["mainshock_longitude"], mainshocks["mainshock_latitude"]),
        crs="EPSG:4326",
    )
    joined = gpd.sjoin(points, plates, how="left", predicate="within")

    missing = joined["PlateName"].isna()
    if missing.any():
        fallback = gpd.sjoin(points.loc[missing], plates, how="left", predicate="intersects")
        joined.loc[missing, "Code"] = fallback["Code"].to_numpy()
        joined.loc[missing, "PlateName"] = fallback["PlateName"].to_numpy()

    joined["plate_code"] = joined["Code"].fillna("UN")
    joined["plate_name"] = joined["PlateName"].fillna("Unassigned")
    joined = pd.DataFrame(joined.drop(columns=["geometry", "index_right", "Code", "PlateName"], errors="ignore"))
    return joined


def add_plate_groups(table: pd.DataFrame, min_count: int = 30) -> tuple[pd.DataFrame, pd.DataFrame]:
    counts = table["plate_name"].value_counts().rename_axis("plate_name").reset_index(name="mainshock_count")
    keep = set(counts.loc[counts["mainshock_count"] >= min_count, "plate_name"])
    grouped = table.copy()
    grouped["plate_group"] = np.where(grouped["plate_name"].isin(keep), grouped["plate_name"], "Other")
    grouped_counts = (
        grouped["plate_group"]
        .value_counts()
        .rename_axis("plate_group")
        .reset_index(name="mainshock_count")
        .sort_values("mainshock_count", ascending=False)
        .reset_index(drop=True)
    )
    grouped_counts["is_other"] = grouped_counts["plate_group"].eq("Other")
    return grouped, grouped_counts

def make_mainshockTable(df_X_list, valid_ids=None):
    """
    for each X_i sequence, create dataframe compatible with assign_plates()
    """
    rows = []
    if valid_ids is None:
        valid_ids = list(range(len(df_X_list)))

    for seq_id, df_X in zip(valid_ids, df_X_list):
        df_X = df_X.copy()
        df_X["datetime"] = pd.to_datetime(df_X["Date"].astype(str) + " " + df_X["Time"].astype(str),format="%Y-%m-%d %H:%M:%S",errors="raise")
        df_X = df_X.sort_values("datetime").reset_index(drop=True)
        main = df_X.iloc[-1]
        rows.append({
            "ID": seq_id,
            "mainshock_time_utc": main["datetime"],
            "mainshock_latitude": float(main["Lat"]),
            "mainshock_longitude": float(main["Lon"]),
            "mainshock_depth_km": float(main["Depth"]),
            "mainshock_mag": float(main["Mag"]),
        })
    return pd.DataFrame(rows)

def fit_plate_groups(df_X_list, valid_ids, plates_path, min_count=30):
    """
    Full pipeline for finding plate groups for Training Data
    (convert X to mainshock table => assign plate => group rare plates to other => return plate_group)
    """
    mainshocks = make_mainshockTable(df_X_list, valid_ids=valid_ids)
    plates = load_plate_polygons(plates_path)

    assigned = assign_plates(mainshocks, plates)
    grouped, counts = add_plate_groups(assigned, min_count=min_count)
    plate_group_by_id = dict(zip(grouped["ID"], grouped["plate_group"]))

    plate_vec = counts["plate_group"].tolist()
    if "Other" not in plate_vec:
        plate_vec.append("Other")
    return plate_group_by_id, plate_vec, grouped, counts, plates

def transform_plate_groups(df_X_list, valid_ids, plates, train_plate_vec):
    """
    Full pipeline for finding plate groups for Test Data
    (convert X to mainshock table => assign plate => group rare plates to other => return plate_group)
    """
    mainshocks = make_mainshockTable(df_X_list, valid_ids=valid_ids)
    assigned = assign_plates(mainshocks, plates)
    known = set(train_plate_vec)

    assigned["plate_group"] = assigned["plate_name"].where(assigned["plate_name"].isin(known),"Other")
    plate_group_by_id = dict(zip(assigned["ID"], assigned["plate_group"]))
    return plate_group_by_id, assigned

def add_plate_onehot(seqdf, plate_group=None, plate_vec=None):
    seqdf = seqdf.copy()
    if plate_group is None:
        plate_group = "Other"
    seqdf["plate_group"] = plate_group
    if plate_vec is not None:
        for p in plate_vec:
            seqdf[f"plate_{p}"] = float(plate_group == p)
    return seqdf


# Mainshock Related Features
def add_diff(seq_df, mainshock_idx):
    """
    - Find temporal & Spatial differences
    """
    seq_df = seq_df.copy()
    main = seq_df.loc[mainshock_idx]

    main_time = main["datetime"]
    seq_df["rel_time_diff"] = (seq_df["datetime"] - main_time).dt.total_seconds() / (60 * 60*24)

    seq_df["rel_mag_diff"] = seq_df["Mag"].astype(float) - main["Mag"]
    seq_df["rel_depth_diff"] = seq_df["Depth"].astype(float) - main["Depth"]

    seq_df["dx"] = main["Lat"] - seq_df["Lat"]
    seq_df["dy"] = main["Lon"] - seq_df["Lon"]
    seq_df["log_flat_dist_km"] = np.log1p(np.sqrt(seq_df["dx"] ** 2 + seq_df["dy"] ** 2))

    seq_df = add_flat_dist(df=seq_df, main_lat=main["Lat"], main_lon=main["Lon"])
    return seq_df


def extract_oneX(seqdf,plate_group=None,plate_vec = None):
    for col in ["Lon", "Lat", "Depth", "Mag"]:
        seqdf[col] = pd.to_numeric(seqdf[col])
    seqdf = encode_time(seqdf)
    mainshock_idx = find_mainshock_idx(seqdf)
    seqdf = add_diff(seqdf, mainshock_idx=mainshock_idx)
    seqdf = add_plate_onehot(seqdf, plate_group=plate_group, plate_vec=plate_vec)
    return seqdf

def scale_seq(df, scaler, medians, scale_cols):
    df = df.copy()
    df[scale_cols] = df[scale_cols].fillna(medians)
    df[scale_cols] = scaler.transform(df[scale_cols])
    return df

def fit_scalar(df, scale_cols):
    train_df = pd.concat(df, axis=0, ignore_index=True)
    medians = train_df[scale_cols].median()
    scaler = StandardScaler().fit(train_df[scale_cols].fillna(medians))
    return scaler, medians