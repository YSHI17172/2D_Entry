#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 25 15:12:50 2019

@author: Yang Shi
"""

import numpy as np
from TwoD_Entry_bvp import Entry

model = Entry(random=True)

print('------------------Random Starting Optimization--------------------------')
optimization_num = 0
op_round_max = 10000  # 成功比例 
mesh_size = 200
sample_all = [] 
while len(sample_all) < op_round_max:
    model.reset()
    for jjj in range(100):
        tf = np.random.randint(12,18)
        # print(tf)
        result = model.get_optimize([tf])
        optimization_num += 1
        if result:
            print('random Starting--', 'round:', len(sample_all), 'step', jjj)
            sample = model.get_sample(result,mesh_size)
            
            #处理多余节点
            if sample.shape[1] > mesh_size:
                while sample.shape[1] > mesh_size:
                    n = sample.shape[1] - mesh_size
                    two = sample.shape[1]//(n + 1)
                    remove = []
                    x = two
                    while x < mesh_size:
                        remove.append(x)
                        x += two
                    sample = np.delete(sample,remove,axis=1)
            
            
            sample_all.append(sample)
            break
print('random Starting--', 'success_ratio:', op_round_max / optimization_num)

np.save("sample_all_Train%d_%d.npy"%(op_round_max,mesh_size),sample_all)



