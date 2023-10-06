import os

# 默认的database地址
defualt_database_path = os.path.dirname(os.path.abspath(__file__))+'/database'

# 粒径统计的筛网组 单位：um
Sieves_group = [[0.5,1,2,3,4,5,6,7,8,9,10,20,30,40,50], 
                [1,2,3,4,5,9],
                [1,2,3,4,5,9],]

# # 双组份粉体混合的权重个数
# weight_number = 50