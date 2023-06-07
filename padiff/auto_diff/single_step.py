
# 这个和 table view 的逻辑差不多，用来找 single_step 时对应的输入
def find_item(self, p_report, net_id, type_):
    tlist = list(filter(lambda x: x.type == type_ and x.net_id == net_id, self.items))
    plist = list(filter(lambda x: x.type == type_ and x.net_id == net_id, p_report.items))
    return tlist[len(plist) - 1]
