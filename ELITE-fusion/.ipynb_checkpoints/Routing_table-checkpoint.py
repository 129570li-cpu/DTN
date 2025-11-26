import numpy as np
import pandas as pd
import math
import fuzzy.FuzzyRouting as FR
from fuzzy import paper_fuzzy as PFR
import random
import traffic as RNTFA
import fuzzy.AExactData as Exact
import fuzzy.FuzzyRules as rules
import fuzzy.FuzzyRouting as FuzR
import Global_Par as Gp

# 每一个SA维护一个路由表,依托各自的孪生网络进行训练
# SDVN中央控制器实例化多个Slave Agent (SA)

class Routing_Table:
    def __init__(self):
        self.table = self.initial_table()
        self.table_PRR, self.table_AD, self.table_HC, self.table_RC = self.get_table()
        self.table_PRR_norm = self.initial_table()
        self.table_AD_norm = self.initial_table()
        self.table_HC_norm = self.initial_table()
        self.table_RC_norm = self.initial_table()
        self.table_BP = self.initial_table()
        self.table_HRF = self.initial_table()
        self.table_LDF = self.initial_table()
        self.table_LBF = self.initial_table()

    # 读文件，初始化
    def get_matrix(self, table_name):
        matrix = []
        # 训练时，读一个全是0的
        with open(table_name, 'r') as file:
            for line in file:
                str = line.split(',')
                if str:
                    row = []
                    for x in str:
                        row.append(float(x))
                    matrix.append(row)
        return matrix

    # 路由表矩阵, dataframe
    # 在选择元素时，先定位列（target），再定位第一维行（current），最后第二维行（adjacent）
    # table[target_area][current_area][next_area]
    # 使用loc[]选择某一行
    def table_config(self, matrix):
        # 读文件获取Q矩阵
        #matrix = self.get_matrix()
        index1 = []
        index2 = []
        column_ = []
        column_len = len(Gp.it_pos)
        row_len = 0
        for it, neibs in Gp.adjacents_comb.items():
            column_.append(it)
            row_len += len(neibs)
            index2.extend(neibs)
            for i in range(0, len(neibs)):
                index1.append(it)
        index_ = [index1, index2]
        # print("row len: ", row_len)
        # print("column_len: ", column_len)
        #----------------------------------
        # np.zeros((row_len, column_len))
        # np.arange(10).reshape(370, 103)
        # matrix
        D = pd.DataFrame(matrix, index=index_, columns=column_)
        #print(D)
        #print(sum(D[47][46]))
        #exit(0)
        return D

    def get_table(self):
        matrix_PRR = self.get_matrix(Gp.file_pdr)
        table_PRR = self.table_config(matrix_PRR)

        matrix_AD = self.get_matrix(Gp.file_ad)
        table_AD = self.table_config(matrix_AD)

        matrix_HC = self.get_matrix(Gp.file_hc)
        table_HC = self.table_config(matrix_HC)

        matrix_RC = self.get_matrix(Gp.file_rc)
        table_RC = self.table_config(matrix_RC)

        return table_PRR, table_AD, table_HC, table_RC

    def initial_table(self):
        index1 = []
        index2 = []
        column_ = []
        column_len = len(Gp.it_pos)
        row_len = 0
        for it, neibs in Gp.adjacents_comb.items():
            column_.append(it)
            row_len += len(neibs)
            index2.extend(neibs)
            for i in range(0, len(neibs)):
                index1.append(it)
        index_ = [index1, index2]
        # a_list = []
        # while len(a_list) < 370*103:
        #     d_int = random.randint(1, 30)
        #     a_list.append(d_int)
        # a = np.array(a_list)
        D = pd.DataFrame(np.zeros((row_len, column_len)), index=index_, columns=column_)
        # #print(D)
        # D.to_csv('table_PRR.csv', sep=',', header=False, index=False)
        # exit(0)
        return D

    # 预处理，归一化操作
    def preprocessing(self):
        # table in (self.table_PRR, self.table_AD, self.table_HC, self.table_RC)
        # 1
        matrix = []
        for cur in Gp.it_pos:
            for neib in Gp.adjacents_comb[cur]:
                row = []
                for des in Gp.it_pos:
                    try:
                        max1 = float(self.table_PRR.loc[(cur, slice(None)), des].max())
                    except Exception:
                        max1 = 0.0
                    if max1 <= 0 or math.log(max1) == 0:
                        row.append(0)
                    else:
                        try:
                            v = float(self.table_PRR.loc[(cur, neib), des])
                        except Exception:
                            v = 0.0
                        row.append(min(1, (math.log(1 + v)) / (math.log(max1))))
                matrix.append(row)
        self.table_PRR_norm = self.table_config(matrix)
        # 2
        matrix = []
        for cur in Gp.it_pos:
            for neib in Gp.adjacents_comb[cur]:
                row = []
                for des in Gp.it_pos:
                    try:
                        max1 = float(self.table_AD.loc[(cur, slice(None)), des].max())
                    except Exception:
                        max1 = 0.0
                    if max1 <= 0 or math.log(max1) == 0:
                        row.append(0)
                    else:
                        try:
                            v = float(self.table_AD.loc[(cur, neib), des])
                        except Exception:
                            v = 0.0
                        row.append(min(1, (math.log(1 + v)) / (math.log(max1))))
                matrix.append(row)
        self.table_AD_norm = self.table_config(matrix)
        # 3
        matrix = []
        for cur in Gp.it_pos:
            for neib in Gp.adjacents_comb[cur]:
                row = []
                for des in Gp.it_pos:
                    try:
                        max1 = float(self.table_HC.loc[(cur, slice(None)), des].max())
                    except Exception:
                        max1 = 0.0
                    if max1 <= 0 or math.log(max1) == 0:
                        row.append(0)
                    else:
                        try:
                            v = float(self.table_HC.loc[(cur, neib), des])
                        except Exception:
                            v = 0.0
                        row.append(min(1, (math.log(1 + v)) / (math.log(max1))))
                matrix.append(row)
        self.table_HC_norm = self.table_config(matrix)
        # 4
        matrix = []
        for cur in Gp.it_pos:
            for neib in Gp.adjacents_comb[cur]:
                row = []
                for des in Gp.it_pos:
                    try:
                        max1 = float(self.table_RC.loc[(cur, slice(None)), des].max())
                    except Exception:
                        max1 = 0.0
                    if max1 <= 0 or math.log(max1) == 0:
                        row.append(0)
                    else:
                        try:
                            v = float(self.table_RC.loc[(cur, neib), des])
                        except Exception:
                            v = 0.0
                        row.append(min(1, (math.log(1 + v)) / (math.log(max1))))
                matrix.append(row)
        self.table_RC_norm = self.table_config(matrix)
        return

    # 基于weight的多路由表融合
    # 融合时注意考虑源和目的是同一个的情况
    def fusion_weight(self):
        self.table_BP = self.table_PRR_norm + self.table_AD_norm + self.table_HC_norm + self.table_RC_norm
        self.table_BP = self.table_BP / 4
        print("table_BP:")
        print(self.table_BP)
        #exit(0)
        return

    def break_point(self, m):
        x1 = 0.2 * m
        x2 = 0.5 * m
        x3 = 0.8 * m
        return x1, x2, x3

    # 基于fuzzy的多路由表融合
    def fusion_fuzzy(self):
        # 1
        matrix = []
        for cur, neibs in Gp.adjacents_comb.items():
            for neib in neibs:
                row = []
                for des in Gp.it_pos:
                    if des in neibs or cur == des:
                        if des == neib:
                            row.append(1)
                        else:
                            row.append(0)
                    else:
                        # 确定分界点
                        # 归一化后已在(0,1]，按论文设置三角函数交点(0.2,0.5,0.8)，不再用m1..m3定标
                        v_prr = self.table_PRR_norm[des][cur][neib]
                        v_ad  = self.table_AD_norm[des][cur][neib]
                        v_rc  = self.table_RC_norm[des][cur][neib]
                        fusion_result = PFR.fuse_hrf(v_prr, v_ad, v_rc)
                        row.append(fusion_result)
                matrix.append(row)
        self.table_HRF = self.table_config(matrix)
        print("table_HRF:")
        print(self.table_HRF)
        # 2
        matrix = []
        for cur, neibs in Gp.adjacents_comb.items():
            for neib in neibs:
                row = []
                for des in Gp.it_pos:
                    if des in neibs or cur == des:
                        if des == neib:
                            row.append(1)
                        else:
                            row.append(0)
                    else:
                        v_ad  = self.table_AD_norm[des][cur][neib]
                        v_hc  = self.table_HC_norm[des][cur][neib]
                        v_prr = self.table_PRR_norm[des][cur][neib]
                        fusion_result = PFR.fuse_ldf(v_ad, v_hc, v_prr)
                        row.append(fusion_result)
                matrix.append(row)
        self.table_LDF = self.table_config(matrix)
        print("table_LDF:")
        print(self.table_LDF)
        # 3
        matrix = []
        for cur, neibs in Gp.adjacents_comb.items():
            for neib in neibs:
                row = []
                for des in Gp.it_pos:
                    if des in neibs or cur == des:
                        if des == neib:
                            row.append(1)
                        else:
                            row.append(0)
                    else:
                        v_rc  = self.table_RC_norm[des][cur][neib]
                        v_hc  = self.table_HC_norm[des][cur][neib]
                        v_ad  = self.table_AD_norm[des][cur][neib]
                        fusion_result = PFR.fuse_lbf(v_rc, v_hc, v_ad)
                        row.append(fusion_result)
                matrix.append(row)
        self.table_LBF = self.table_config(matrix)
        print("table_LBF:")
        print(self.table_LBF)
        return

