import os
from pathlib import Path

current_dir = os.path.split(os.path.realpath(__file__))[0]
project_dir = current_dir

div_data_dir = project_dir + '/数据及处理/divdata'
Path(div_data_dir).mkdir(parents=True, exist_ok=True)
source_data_dir = project_dir + '/数据及处理/sourcedata'
Path(source_data_dir).mkdir(parents=True, exist_ok=True)

output_dir = project_dir + '/output'
Path(output_dir).mkdir(parents=True, exist_ok=True)
div_store_model_dir = project_dir + '/模型保存/div'
Path(div_store_model_dir).mkdir(parents=True, exist_ok=True)
source_store_model_dir = project_dir + '/模型保存/source'
Path(source_store_model_dir).mkdir(parents=True, exist_ok=True)


merge_dir = project_dir




