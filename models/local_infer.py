from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import helpers.load_data as ld
import helpers.evaluate as eh
import helpers.baseline_models as bsm
import add_func as af

# Define the Path
TEST_DATA_PATH = ""   # paste the local test folder (with X_<id>.csv and optionally have T_1,T_2...)
SCALER_PATH = "models/saved_models/cache/scaler.pkl"
MODEL_PATH = "models/saved_models/baseline_models.pkl" 
OUT_DIR = Path("inference_out")
OUT_DIR.mkdir(exist_ok=True)

WINDOWS = {"T1": (0.0, 1.0),
           "T2": (1.0, 3.0),
           "T3": (3.0, 7.0)}
min_mag = 3.0
max_obs = None
MAX_LEN = 300          
MODEL_NAME = "XGBoost" 
RECENT_K = 50

# Load test Seq
testX_raw, testY_raw, test_ids = ld.load_raw_data(TEST_DATA_PATH, min_mag, max_obs, seed=123)
print("# test sequences:", len(testX_raw))

# Build Features 
testX_cap = [df.tail(MAX_LEN).reset_index(drop=True) for df in testX_raw]
testX_array, testY_array, test_feature_cols = af.build_XY_test(testX_cap, testY_raw, scaler_path=str(SCALER_PATH))

# Load trained trees and predict
models = bsm.load_baseline_models(MODEL_PATH)
preds  = bsm.predict_baseline_models(models, testX_array, recent_k=RECENT_K)
pred_tab = preds[MODEL_NAME]

# Decode each sequence to physical units
rows = []
for i, seq_id in enumerate(test_ids):
    main = af.mainshock_ref(testX_raw[i])              
    p = pred_tab[i].reshape(3, 2)
    for k, key in enumerate(("T1", "T2", "T3")):
        w0, w1 = WINDOWS[key]
        norm_t = float(np.clip(p[k, 1], 0.0, 1.0))
        t_pred = main["datetime"] + pd.to_timedelta(w0 + norm_t * (w1 - w0), unit="D")
        rows.append({"model": MODEL_NAME, "ID": seq_id, "T": k + 1,
                     "predicted_time": t_pred.strftime("%Y%m%d%H%M%S"),
                     "predicted_mag": round(main["mag"] + float(p[k, 0]), 1)})
results_df = pd.DataFrame(rows)[["model","ID","T","predicted_time","predicted_mag"]]
results_df.to_csv(OUT_DIR / "xgb_Own_predictions.csv", index=False)
print(results_df.head(10))

# Evaluate to ground truth if provided:
# metrics = eh.evaluate_predictions(results_df, test_ids, testY_raw, model_name=MODEL_NAME)
# print(eh.summarize_metrics(metrics))