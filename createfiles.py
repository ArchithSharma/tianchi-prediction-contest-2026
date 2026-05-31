import pandas as pd
import zipfile
from pathlib import Path

# -----------------------------
# Load predictions
# -----------------------------
preds = pd.read_csv("baseline_predictions.csv")

# Keep only XGBoost predictions
preds = preds[preds["model"] == "XGBoost"].copy()

# -----------------------------
# Load catalog
# -----------------------------
with zipfile.ZipFile("test_eq_data.zip") as z:
    catalog = pd.read_csv(z.open("test_eq_catalog.csv"))

# Output directory
outdir = Path("submission_files")
outdir.mkdir(exist_ok=True)

# -----------------------------
# Create submission files
# -----------------------------
for event_id in sorted(preds["ID"].unique()):

    event = catalog.iloc[event_id]

    mainshock_time = (
        f"{int(event.Year):04d}"
        f"{int(event.Month):02d}"
        f"{int(event.Day):02d}"
        f"{int(event.Hour):02d}"
        f"{int(event.Minute):02d}"
        f"{int(event.Second):02d}"
    )

    lon = event.Lon
    lat = event.Lat
    mag = event.Mag

    event_preds = preds[preds["ID"] == event_id]

    # ----- T1 + T2 file -----
    rows_t12 = []

    for T in [1, 2]:
        row = event_preds[event_preds["T"] == T].iloc[0]

        pred_mag = round(float(row["predicted_mag"]), 1)

        pred_time = str(int(row["predicted_time"]))
        pred_time_hour = pred_time[:10]  # YYYYMMDDHH

        rows_t12.append([
            mainshock_time,
            lon,
            lat,
            mag,
            f"{pred_mag:.1f} (Ms)",
            pred_time_hour
        ])

    pd.DataFrame(rows_t12).to_csv(
        outdir / f"{mainshock_time}-T1-T2.csv",
        header=False,
        index=False
    )

    # ----- T3 file -----
    row = event_preds[event_preds["T"] == 3].iloc[0]

    pred_mag = round(float(row["predicted_mag"]), 1)

    pred_time = str(int(row["predicted_time"]))
    pred_time_hour = pred_time[:10]

    pd.DataFrame([[
        mainshock_time,
        lon,
        lat,
        mag,
        f"{pred_mag:.1f} (Ms)",
        pred_time_hour
    ]]).to_csv(
        outdir / f"{mainshock_time}-T3.csv",
        header=False,
        index=False
    )

print(f"Created {len(preds['ID'].unique()) * 2} submission files.")
