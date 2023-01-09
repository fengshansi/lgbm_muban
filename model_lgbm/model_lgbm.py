from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor, Booster, early_stopping, log_evaluation
import os
import pandas as pd
import numpy as np
from lgbm_config import *
from lgbm_utils import *

import sys
from pathlib import Path
# 本文件目录
real_dir = Path(__file__).resolve().parent
# 需要转str才行
sys.path.append(str(real_dir.parent))
import config


current_dir = os.path.split(os.path.realpath(__file__))[0]
iterations = stop_iteration-start_iteration

init_model = None if start_iteration == 0 else store_str % start_iteration




lgbm_model = LGBMRegressor(
    boosting_type='gbdt',  # 设置提升类型，默认gbdt,传统梯度提升决策树
    objective='regression',  # 目标函数mae
    metric='mae',  # 评估函数
    max_depth=7,  # 树的深度 按层，默认-1
    n_jobs=16,  # 线程数量 为了获得最佳速度，请将其设置为实际 CPU 内核的数量，而不是线程数
    n_estimators=iterations,  # 默认100,别名=num_iteration,
    num_leaves=126,  # 在一棵树中叶子节点数,默认31
    learning_rate=0.04,  # 学习速率
    # bagging_freq=4,  # k 意味着每 k 次迭代执行bagging
    verbose=-1,  # <0 显示致命的, =0 显示错误 (警告), >0 显示信息
    device='cpu',
    min_child_samples=65,
    random_state=2022
)



# 读取数据
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



# 获得使用的分类特征
current_cat_feature = list(
    set(all_cat_features) & set(use_feature))


# 进行训练
if iterations != 0:

    def save_model_callback(env):
        iteration = env.iteration
        begin_iteration = env.begin_iteration
        end_iteration = env.end_iteration
        # print(env)
        if iteration != 0 and iteration % store_interval == 0:
            # 这里不需要model.booster_.save_model 本身这里的model就是booster对象
            env.model.save_model(store_str % iteration)

    lgbm_model.fit(
        train_input, train_label,
        eval_set=[(train_input, train_label)],
        categorical_feature=current_cat_feature,
        init_model=init_model,
        # early_stopping针对的是eval_set
        # log_evaluation 打印 用10周期
        callbacks=[early_stopping(50, first_metric_only=True),
                   save_model_callback, log_evaluation(period=100, show_stdv=True)]
    )
    # 从regeressor中提取booster模型 regeressor应该是一种包装
    lgbm_model_booster = lgbm_model.booster_

    lgbm_model_booster.save_model(store_str % stop_iteration)

else:
    lgbm_model_booster = Booster(model_file=init_model)


# 获得重要性
get_lightgbm_importance(
    lgbm_model_booster, config.output_dir +"/importance_lightgbm.csv")



# 获得预测结果并写入
test_label = lgbm_model_booster.predict(test_input)
test_label = pd.DataFrame({label_name: test_label})
test_label.to_csv(config.output_dir+'/submission_raw_lightgbm.csv', index=False)
