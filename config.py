import os
import numpy as np

# 默认的database地址
defualt_database_path = os.path.dirname(os.path.abspath(__file__))+'/database'

# 粒径统计的筛网组 单位：um
Sieves_group = [np.logspace(-2, 2, num=37).tolist(),
                [0.5,1,2,3,4,5,6,7,8,9,10,20,30,40,50], 
                [1,2,3,4,5,9],
                ]