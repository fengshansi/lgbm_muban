from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor, Booster, early_stopping, log_evaluation
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
from pprint import pprint
import optuna
import subprocess
from optuna.integration import LightGBMPruningCallback

from lgbm_opt_config import *
from lgbm_opt_utils import *

import sys
from pathlib import Path
# 本文件目录
real_dir = Path(__file__).resolve().parent
# 需要转str才行
sys.path.append(str(real_dir.parent))
import config



current_dir = os.path.split(os.path.realpath(__file__))[0]



train_input = pd.read_pickle(train_input_pkl)
test_input = pd.read_pickle(test_input_pkl)
train_label = pd.read_pickle(train_label_pkl)

# 数据类型修改
# print(train_input.dtypes.to_dict())


# 用feature_filter进行特征过滤
use_feature = feature_filter.filter(train_input.columns.to_list())
print(f'{len(use_feature)+1} features')
train_input = train_input[use_feature]
test_input = test_input[use_feature]






def objective(trial):
    
    # cat_feature
    all_cat_features = [
        trial.suggest_categorical('cat1',['None', '分类特征1']),
        trial.suggest_categorical('cat2',['None', '分类特征2'])
    ]
    all_cat_features = [x for x in all_cat_features if x != 'None']

    lgbm_model = LGBMRegressor(
        boosting_type='gbdt',  # 设置提升类型，默认gbdt,传统梯度提升决策树
        # objective=trial.suggest_categorical('objective',['rmse', 'regression']),# 目标函数
        objective='regression', 
        metric='mae',  # 评估函数
        max_depth=trial.suggest_int('max_depth', 4, 12),  # 树的深度 按层，默认-1
        n_jobs=16, # 线程数量 为了获得最佳速度，请将其设置为实际 CPU 内核的数量，而不是线程数
        n_estimators=trial.suggest_int('iterations', 800, 1600),  # 默认100,别名=num_iteration,
        # n_estimators=500,  # 默认100,别名=num_iteration,
        num_leaves=trial.suggest_int('num_leaves', 32, 256),  # 在一棵树中叶子节点数,默认31
        learning_rate=trial.suggest_float('learning_rate', 0.01, 0.1, step=0.01),  # 学习速率
        # bagging_freq=4,  # k 意味着每 k 次迭代执行bagging
        verbose=-1,  # <0 显示致命的, =0 显示错误 (警告), >0 显示信息
        device='cpu',
        min_child_samples=trial.suggest_int('min_child_samples', 5, 100),
        random_state=2022
    )

    lgbm_model.fit(
        train_input, train_label,
        eval_set=[(train_input, train_label)],
        # early_stopping针对的是eval_set
        # log_evaluation 打印 用10周期
        categorical_feature=all_cat_features,
        callbacks=[early_stopping(50, first_metric_only=True),
                   log_evaluation(period=100, show_stdv=True)]
    )
    # 从regeressor中提取booster模型 regeressor应该是一种包装
    lgbm_model_booster = lgbm_model.booster_
 

    test_label = lgbm_model_booster.predict(test_input)
    test_label = pd.DataFrame({label_name: test_label})
    test_label.to_csv(config.output_dir+'/submission_raw_lightgbm.csv', index=False)

    # 测试
    p = subprocess.Popen( ['python',config.merge_dir+'/merge_div.py'],stdout=subprocess.PIPE)
    p.wait()
    test_result = float(p.stdout.read())
 
    return test_result
    

if __name__ == '__main__':

    study_name = "test"
    storage_name = f"sqlite:///{current_dir}/{study_name}.db"
    study = optuna.create_study(
        study_name=study_name, direction=direction, storage=storage_name, load_if_exists=True)
    # 如果要加载以前运行的db文件，注释掉下面这句
    study.optimize(objective, n_trials=n_trials)
    print(study.best_params)
    print(study.best_trial)
    print(study.best_trial.value)
    
    # 原始的使用方法
    # optuna.visualization.plot_param_importances(study).write_image(current_dir+'/optuna_plot_param_importances.jpg')
    # 便捷调用多个绘画函数
    plot_func_name_list = ['plot_param_importances','plot_optimization_history','plot_slice']
    for plot_func_name in plot_func_name_list:
        plot_func = getattr(optuna.visualization,plot_func_name)
        plot_func(study).write_image(current_dir+f'/{plot_func_name}.jpg')
   
 