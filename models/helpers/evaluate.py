import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path


def plot_prediction_results(results_df, test_ids, testX_raw, testY_raw, file_path, file_name, bg_path, n_plot=20):
    fig, axes = plt.subplots(4, 5, figsize=(20, 12))

    for i in range(min(n_plot, len(test_ids))):
        ax = axes[i // 5, i % 5]
        seq_id = test_ids[i]
        bg_file = Path(bg_path) / f"{seq_id}.csv"

        # plot background events
        df = pd.read_csv(bg_file)
        ax.scatter(pd.to_datetime(df["Date"] + " " + df["Time"]),pd.to_numeric(df["Mag"]),s=10,c="gray",label="Background Events")

        # plot mainshock and T1/T2/T3 windows
        df = testX_raw[i].copy()
        event_time = pd.to_datetime(df["Date"] + " " + df["Time"])
        mainshock_time = event_time.iloc[-1]

        ax.axvline(mainshock_time,color="red",linestyle="--",label="Mainshock")

        ax.fill_betweenx([0, 10],mainshock_time + pd.to_timedelta(0, unit="D"),
                         mainshock_time + pd.to_timedelta(1, unit="D"),
                         color="orange",alpha=0.3,
                         label="T1 Window")

        ax.fill_betweenx([0, 10],mainshock_time + pd.to_timedelta(1, unit="D"),
                         mainshock_time + pd.to_timedelta(3, unit="D"),
                         color="yellow",alpha=0.3,
                         label="T2 Window")

        ax.fill_betweenx([0, 10],mainshock_time + pd.to_timedelta(3, unit="D"),
                         mainshock_time + pd.to_timedelta(7, unit="D"),
                         color="green",alpha=0.3,label="T3 Window")

        # plot true aftershocks
        df = testY_raw[i].copy()
        ax.scatter(pd.to_datetime(df["Date"] + " " + df["Time"]),
                   pd.to_numeric(df["Mag"]),s=50,c="blue",label="True")

        # plot predicted aftershocks
        pred_df = results_df[results_df["ID"] == seq_id].copy()

        pred_time = pd.to_datetime(pred_df["predicted_time"].astype(str).str.replace("'", "", regex=False),
                                   format="%Y%m%d%H%M%S",errors="coerce")

        ax.scatter(pred_time,pred_df["predicted_mag"],s=100,c="cyan",marker="X",label="Prediction")

        ax.tick_params(axis="x", rotation=45)
        ax.set_ylabel("Mag")
        ax.set_xlabel("Date")
        ax.set_ylim(0, 10)
        ax.legend(loc="upper right", fontsize=3)
    plt.tight_layout()
    Path("fig").mkdir(parents=True, exist_ok=True)
    plt.savefig(f"{file_path}/prediction_results_{file_name}.pdf")



def evaluate_predictions(results_df, test_ids, testY_raw, model_name=None, save_path=None):
    results_df = results_df.copy()

    if model_name is not None and "model" in results_df.columns:
        results_df = results_df[results_df["model"] == model_name].copy()

    metric_rows = []

    for i, seq_id in enumerate(test_ids):
        pred_df = results_df[results_df["ID"] == seq_id].copy()
        true_df = testY_raw[i].copy()
        if len(pred_df) == 0:continue

        if "T" not in true_df.columns:true_df["T"] = [1, 2, 3]

        pred_df["pred_time_dt"] = pd.to_datetime(pred_df["predicted_time"].astype(str).str.replace("'", "", regex=False),
                                                 format="%Y%m%d%H%M%S",errors="coerce")

        true_df["true_time_dt"] = pd.to_datetime(true_df["Date"].astype(str) + " " + true_df["Time"].astype(str),
                                                 format="%Y-%m-%d %H:%M:%S",errors="coerce")

        true_df["Mag"] = pd.to_numeric(true_df["Mag"], errors="coerce")
        pred_df["predicted_mag"] = pd.to_numeric(pred_df["predicted_mag"], errors="coerce")

        for T in [1, 2, 3]:
            p_rows = pred_df[pred_df["T"] == T]
            y_rows = true_df[true_df["T"] == T]

            if len(p_rows) == 0 or len(y_rows) == 0:
                continue
            for _, p in p_rows.iterrows():
                y = y_rows.iloc[0]
                signed_time_err_hours = (p["pred_time_dt"] - y["true_time_dt"]).total_seconds() / 3600

                signed_mag_err = p["predicted_mag"] - y["Mag"]

                metric_rows.append({
                    "model": p["model"] if "model" in pred_df.columns else (
                        model_name if model_name is not None else "model"),
                    "ID": seq_id,
                    "T": T,

                    "pred_time": p["pred_time_dt"],
                    "true_time": y["true_time_dt"],
                    "signed_time_err_hours": signed_time_err_hours,
                    "time_abs_err_hours": abs(signed_time_err_hours),

                    "pred_mag": p["predicted_mag"],
                    "true_mag": y["Mag"],
                    "signed_mag_err": signed_mag_err,
                    "mag_abs_err": abs(signed_mag_err)})

    metrics_df = pd.DataFrame(metric_rows)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_df.to_csv(save_path, index=False)

    return metrics_df

def summarize_metrics(metrics_df, save_dir=None, prefix="metrics"):
    summary_by_T = (
        metrics_df
        .groupby(["model", "T"])
        .agg(
            time_mae_hours=("time_abs_err_hours", "mean"),
            time_median_abs_err_hours=("time_abs_err_hours", "median"),
            mag_mae=("mag_abs_err", "mean"),
            mag_median_abs_err=("mag_abs_err", "median"),
        )
        .reset_index()
    )
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        summary_by_T.to_csv(save_dir / f"{prefix}_by_T.csv", index=False)

    return summary_by_T