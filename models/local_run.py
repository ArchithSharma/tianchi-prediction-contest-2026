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

import add_func as af

# File path

PLATES_PATH     = "plates/plate_mainshock_analysis/tectonicplates-master/GeoJSON/PB2002_plates"
TRAIN_DATA_PATH = "eq-data/cleaned_data/training_trajectories_full" #Must be imported from google drive
TEST_DATA_PATH  = "eq-data/cleaned_data/test_seq" #Must be imported from google drive

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
