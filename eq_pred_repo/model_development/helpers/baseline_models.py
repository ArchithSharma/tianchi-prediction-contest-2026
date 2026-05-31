import pickle
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.multioutput import MultiOutputRegressor

import lightgbm as lgbm
import xgboost as xgb


def seq_to_tabular(X, recent_k=50):
    """
    Convert one variable-length sequence array [seq_len, n_features]
    into one fixed-length tabular vector.

    X is already engineered/scaled from build_XY_train/build_XY_test.
    """
    X = np.asarray(X, dtype=np.float32)

    if X.ndim != 2:
        raise ValueError(f"Expected X with shape [seq_len, n_features], got {X.shape}")

    seq_len = np.array([len(X)], dtype=np.float32)
    last_event = X[-1]
    mean_event = X.mean(axis=0)
    std_event = X.std(axis=0)
    max_event = X.max(axis=0)
    min_event = X.min(axis=0)

    recent = X[-recent_k:] if len(X) > recent_k else X
    recent_mean = recent.mean(axis=0)
    recent_std = recent.std(axis=0)

    return np.concatenate([seq_len,last_event,mean_event,
                           std_event,max_event,min_event,
                           recent_mean,recent_std]).astype(np.float32)


def make_tabular_X(X_list, recent_k=50):
    """
    Convert list of variable-length X arrays into fixed tabular matrix.
    """
    return np.vstack([seq_to_tabular(X, recent_k=recent_k) for X in X_list])


def make_tabular_Y(Y_list):
    """
    Convert list of Y arrays [3, 2] into matrix [n_sequences, 6].
    Current target convention:
        Y[:, 0] = delta_mag
        Y[:, 1] = normalized time within T window
    """
    return np.vstack([np.asarray(Y, dtype=np.float32).reshape(1, -1)for Y in Y_list])


def build_baseline_models(n_trees=500, random_state=12345):
    """
    Return baseline regression models.
    Output dimension is 6:
        [T1_delta_mag, T1_norm_time,
         T2_delta_mag, T2_norm_time,
         T3_delta_mag, T3_norm_time]
    """

    models = {}
    models["LightGBM"] = MultiOutputRegressor(
        lgbm.LGBMRegressor(
            n_estimators=n_trees,
            learning_rate=0.03,
            num_leaves=15,
            max_depth=5,
            min_child_samples=10,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=random_state,
            verbose=-1,
        )
    )

    models["XGBoost"] = MultiOutputRegressor(
        xgb.XGBRegressor(
            n_estimators=n_trees,
            learning_rate=0.03,
            max_depth=3,
            min_child_weight=5,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            objective="reg:squarederror",
            random_state=random_state,
            n_jobs=-1,
        )
    )
    return models


def fit_baseline_models(
    X_arrays,
    Y_arrays,
    model_names=("LightGBM",),
    recent_k=50,
    n_trees=500,
    random_state=12345,
):
    """
    Fit selected baseline models.
    """

    X_train_tab = make_tabular_X(X_arrays, recent_k=recent_k)
    Y_train_tab = make_tabular_Y(Y_arrays)

    all_models = build_baseline_models(
        n_trees=n_trees,
        random_state=random_state,
    )

    fitted = {}

    for name in model_names:
        print(f"\nTraining {name}...")
        model = all_models[name]
        model.fit(X_train_tab, Y_train_tab)
        fitted[name] = model

        train_pred = model.predict(X_train_tab)
        train_mae = mean_absolute_error(Y_train_tab, train_pred)
        print(f"{name} train encoded MAE: {train_mae:.6f}")

    return fitted


def predict_baseline_models(models,testX_array,recent_k=50):
    """
    Predict encoded outputs for test data.
    """
    X_test_tab = make_tabular_X(testX_array, recent_k=recent_k)
    preds = {}
    for name, model in models.items():
        print(f"Predicting {name}...")
        pred = model.predict(X_test_tab)
        pred[:, [1, 3, 5]] = np.clip(pred[:, [1, 3, 5]], 0.0, 1.0)
        preds[name] = pred
    return preds


def fit_predict_baselines(X_arrays,Y_arrays,testX_array,model_names=("LightGBM",),recent_k=50,n_trees=500,random_state=12345):
    """
    Convenience wrapper:
        train baselines -> predict test set.
    """

    models = fit_baseline_models(
        X_arrays=X_arrays,
        Y_arrays=Y_arrays,
        model_names=model_names,
        recent_k=recent_k,
        n_trees=n_trees,
        random_state=random_state,
    )

    preds = predict_baseline_models(
        models=models,
        testX_array=testX_array,
        recent_k=recent_k,
    )

    return models, preds


def save_baseline_models(models, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "wb") as f:
        pickle.dump(models, f)


def load_baseline_models(path):
    with open(path, "rb") as f:
        return pickle.load(f)