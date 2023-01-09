import os

import sys
from pathlib import Path
# 本文件目录
real_dir = Path(__file__).resolve().parent
# 需要转str才行
sys.path.append(str(real_dir.parent))
import config
from utils import Feature_Filter

current_dir = os.path.split(os.path.realpath(__file__))[0]




# 数据集===============================================================
# opt只支持div
train_input_pkl = config.div_data_dir+'占位'
test_input_pkl = config.div_data_dir+'占位'
train_label_pkl = config.div_data_dir+'占位'
store_str = config.div_store_model_dir+'占位'
# 数据集===============================================================




# 训练参数==================================================

n_trials = 10
direction='maximize'

label_name = '占位'
# 下面这几个参数opt不需要
# store_interval = 100
# # 下面两个相同进入测试模式
# start_iteration = 0
# stop_iteration = 1500

# all_cat_features opt里调  
# all_cat_features = ['占位']
# 训练参数==================================================


# 特征选取====================================================
drop_list = [
    '占位'
]
feature_filter = Feature_Filter("drop", drop_list)


# pick_list = [
#     '占位'
# ]
# feature_filter = Feature_Filter("pick", drop_list)

# 特征选取====================================================