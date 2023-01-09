import pandas as pd

def get_lightgbm_importance(bst, importance_file):
    # 按importance_gain排序
    df_important = (
        pd.DataFrame({
            'feature_name': bst.feature_name(),
            'importance_gain': bst.feature_importance(importance_type='gain'),
            'importance_split': bst.feature_importance(importance_type='split'),
        })
        .sort_values('importance_gain', ascending=False)
        .reset_index(drop=True)
    )

    # 存储
    df_important.to_csv(importance_file, index=False)

