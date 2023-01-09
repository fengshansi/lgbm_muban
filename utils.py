class Feature_Filter():
    def __init__(self, filter_mode, filter_list):
        assert filter_mode in [
            "drop", "pick"], "filter_mode must be drop or pick"
        # 必须传入list 例如如果只pick一列 不能用df = df['uid']，要用df = df[['uid']]
        # 前者是Series，后者虽然也是只有一列，但是是DataFrame
        assert type(filter_list) == list, "filter_list must be list"
        self.filter_mode = filter_mode
        self.filter_list = filter_list

        # 初始化函数指针指向
        if self.filter_mode == "drop":
            self.filter_point = self.drop_filter
        elif self.filter_mode == "pick":
            self.filter_point = self.pick_filter

    def drop_filter(self, all_feature_list):
        return [x for x in all_feature_list if x not in self.filter_list]

    def pick_filter(self, all_feature_list):
        return self.filter_list

    def filter(self, all_feature_list):
        # 调用函数指针
        return self.filter_point(all_feature_list)