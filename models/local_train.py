from pathlib import Path
import numpy as np
import pandas as pd
import torch
import joblib
import helpers.load_data as ld
import helpers.evaluate as eh
import helpers.baseline_models as bsm
import add_func as af

# Local version of code_notebook.ipynb

# File path
PLATES_PATH     = "models/plate_mainshock_analysis/tectonicplates-master/GeoJSON/PB2002_plates"
TRAIN_DATA_PATH = "cleaned_data" 
TEST_DATA_PATH  = "cleaned_data/test_seq"

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


cache_name = "models/saved_models/cache"
cache_dir = Path(f"{cache_name}")
cache_dir.mkdir(parents=True, exist_ok=True)
max_obs = None
min_mag = 3.0
scaler_path = cache_dir / "scaler.pkl"
cache_path = cache_dir / f"features_minmag{min_mag}_maxobs{max_obs}.joblib"

df_X_list, df_Y_list, valid_ids = ld.load_raw_data(TRAIN_DATA_PATH, min_mag, max_obs, seed=123)
testX_raw, testY_raw, test_ids   = ld.load_raw_data(TEST_DATA_PATH,  min_mag, max_obs, seed=123)
print("# train:", len(df_X_list), "# test:", len(testX_raw))


PRE_CAP = 300   # keep last 300 events per sequence; mainshock is the last row
df_X_cap     = [df.tail(PRE_CAP).reset_index(drop=True) for df in df_X_list]
testX_raw_cap = [df.tail(PRE_CAP).reset_index(drop=True) for df in testX_raw]

import gc
del df_X_list 
gc.collect()
print("pre-capped. max train rows now:", max(len(d) for d in df_X_cap))

X_arrays, Y_arrays, feature_cols = af.build_XY_train(df_X_cap, df_Y_list, scaler_path=str(scaler_path))
print("train dim:", X_arrays[0].shape[1])

testX_array, testY_array, test_feature_cols = af.build_XY_test(testX_raw_cap, testY_raw, scaler_path=str(scaler_path))
assert feature_cols == test_feature_cols, "feature mismatch"

cache = {"valid_ids": valid_ids, "test_ids": test_ids,
         "X_arrays": X_arrays, "Y_arrays": Y_arrays,
         "testX_array": testX_array, "testY_array": testY_array,
         "testY_raw": testY_raw, "testX_raw": testX_raw,   # full raw kept for decode/plot
         "feature_cols": feature_cols, "test_feature_cols": test_feature_cols,
         "min_mag": min_mag, "max_obs": max_obs, "scaler_path": str(scaler_path)}
gc.collect()
joblib.dump(cache, cache_path, compress=0)
print("saved", cache_path)


save_path = Path("/models/saved_models/")
save_path.mkdir(parents=True, exist_ok=True)

cache = joblib.load(cache_path)
valid_ids = cache["valid_ids"]

testX_raw, testY_raw, test_ids = cache["testX_raw"], cache["testY_raw"], cache["test_ids"]
X_arrays = cache["X_arrays"]
Y_arrays = cache["Y_arrays"]
testX_array = cache["testX_array"]
testY_array = cache["testY_array"]

feature_cols = cache["feature_cols"]
test_feature_cols = cache["test_feature_cols"]

from neuralmodel import NeuralModel

torch.backends.cudnn.enabled = True
MAX_LEN = 300
X_arrays   = [x[-MAX_LEN:] for x in X_arrays]
testX_array = [x[-MAX_LEN:] for x in testX_array]   # same cap on test
print("after cap — train max:", max(x.shape[0] for x in X_arrays),
      "| test max:", max(x.shape[0] for x in testX_array))

# validation split (from training data)
val_frac = 0.15
n = len(X_arrays)
rng = np.random.default_rng(0)
perm = rng.permutation(n)
n_val = int(val_frac * n)
val_idx, tr_idx = perm[:n_val], perm[n_val:]

X_tr  = [X_arrays[i] for i in tr_idx]
Y_tr  = [Y_arrays[i] for i in tr_idx]
X_val = [X_arrays[i] for i in val_idx]
Y_val = [Y_arrays[i] for i in val_idx]
print(f"train: {len(X_tr)}  val: {len(X_val)}  test: {len(testX_array)}")


device_name = "cuda" if torch.cuda.is_available() else "cpu"
print(device_name)
lr = 1e-3
epochs = 150
hidden_size = 256
SAVE_MODEL_PATH = Path("/models/saved_models/neural_model.pt")
SAVE_EVAL_PATH = Path("/models/saved_models")
model = NeuralModel(input_size=len(feature_cols),hidden_size=hidden_size,output_size=2)
model.fit(X_tr,Y_tr,device=device_name,lr=lr,epochs=epochs,  save_path = SAVE_MODEL_PATH)

print("Saved model to", SAVE_MODEL_PATH)

def val_mae(model, X_list, Y_list, device):
    model.eval()
    # window widths in HOURS, in T1,T2,T3 order
    win_hours = np.array([(WINDOWS[k][1] - WINDOWS[k][0]) * 24.0 for k in ("T1","T2","T3")])  # [24,48,96]
    mag_err, time_err = [], []
    with torch.no_grad():
        for x, y in zip(X_list, Y_list):
            X = torch.tensor(x, dtype=torch.float32, device=device).unsqueeze(0).contiguous()
            p = model.forward(X).squeeze(0).cpu().numpy() 
            p_time = np.clip(p[:, 1], 0.0, 1.0)
            y_time = y[:, 1]
            mag_err.append(np.abs(p[:, 0] - y[:, 0]))
            time_err.append(np.abs(p_time - y_time) * win_hours)
    mag_err  = np.array(mag_err)
    time_err = np.array(time_err)
    print("validation MAE (physical units):")
    for t in range(3):
        print(f"  T{t+1}:  mag {mag_err[:, t].mean():.4f}   time {time_err[:, t].mean():.4f} h")
    print(f"  overall:  mag {mag_err.mean():.4f}   time {time_err.mean():.4f} h")
    return mag_err, time_err
    
val_mae(model, X_tr, Y_tr, device_name)
val_mae(model, X_val, Y_val, device_name)

results = []
bg_path = "/cleaned_data/test_data"

for i in range(len(testX_array)):
    pred_df = model.predict(df_X=testX_array[i],df_X_raw=testX_raw[i],device=device_name)
    pred_df["ID"] = test_ids[i]
    results.append(pred_df)
results_df = pd.concat(results, axis=0, ignore_index=True)
results_df["model"] = "NeuralModel"
results_df = results_df[["model", "ID", "T", "predicted_time", "predicted_mag"]]
results_df.to_csv(save_path / "neural_predictions.csv", index=False)
neural_metrics_df = eh.evaluate_predictions(results_df=results_df,
                                         test_ids=test_ids,
                                         testY_raw=testY_raw,
                                         model_name="NeuralModel",
                                         save_path=save_path / "neural_metrics_by_prediction.csv")
neural_summary_by_T = eh.summarize_metrics(neural_metrics_df,
                                        save_dir=save_path,
                                        prefix="neural")
eh.plot_prediction_results(results_df=results_df,
                        test_ids=test_ids,
                        testX_raw=testX_raw,
                        testY_raw=testY_raw,
                        file_path=save_path,
                        file_name="neural",
                        bg_path = bg_path,
                        n_plot=20)

print(neural_summary_by_T)

baseline_models, baseline_preds = bsm.fit_predict_baselines(X_arrays=X_arrays,
                                                        Y_arrays=Y_arrays,
                                                        testX_array=testX_array,
                                                        model_names=("LightGBM", "XGBoost"),
                                                        recent_k=50,
                                                        n_trees=500)

bsm.save_baseline_models(baseline_models,
                     save_path / "baseline_models.pkl")

baseline_results = []

for model_name, pred_tab in baseline_preds.items():
    for i in range(len(testX_array)):
        pred_i = pred_tab[i].reshape(3, 2)

        pred_df = decode_predictions(pred_i, testX_raw[i])
        pred_df["ID"] = test_ids[i]
        pred_df["model"] = model_name

        baseline_results.append(pred_df)

baseline_results_df = pd.concat(baseline_results, axis=0, ignore_index=True)
baseline_results_df = baseline_results_df[["model", "ID", "T", "predicted_time", "predicted_mag"]]

baseline_results_df.to_csv(save_path / "baseline_predictions.csv", index=False)

baseline_metrics_df = eh.evaluate_predictions(results_df=baseline_results_df,
                                           test_ids=test_ids,
                                           testY_raw=testY_raw,
                                           model_name=None,
                                           save_path=save_path / "baseline_metrics_by_prediction.csv")

baseline_summary_by_T = eh.summarize_metrics(baseline_metrics_df,
                                          save_dir=save_path,
                                          prefix="baseline")
for model_name in baseline_results_df["model"].unique():
    eh.plot_prediction_results(
        results_df=baseline_results_df[
            baseline_results_df["model"] == model_name
        ].copy(),
        test_ids=test_ids,
        testX_raw=testX_raw,
        testY_raw=testY_raw,
        file_name=model_name.lower(),
        file_path=save_path,
        n_plot=20,
        bg_path = bg_path
    )
print(baseline_summary_by_T)