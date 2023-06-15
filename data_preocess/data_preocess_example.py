
from catboost import CatBoostRegressor
import zipfile
from numpy import sqrt
from sklearn.preprocessing import StandardScaler, QuantileTransformer, KBinsDiscretizer, LabelEncoder, MinMaxScaler, PowerTransformer
import os
import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import roc_auc_score, auc, roc_curve, accuracy_score, f1_score, mean_absolute_error
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
import pickle
import logging
from timeit import default_timer as timer


import sys
from pathlib import Path
# 本文件目录
real_dir = Path(__file__).resolve().parent
# 需要转str才行
sys.path.append(str(real_dir.parent))
import config



current_dir = os.path.split(os.path.realpath(__file__))[0]





