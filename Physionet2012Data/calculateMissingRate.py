#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 18 14:53:17 2018

@author: lyh
"""
import sys
sys.path.append("..")
import numpy as np
import os
from Physionet2012Data import readData
def f():
    dt_train=readData.ReadPhysionetData(os.path.join("../set-a","train"), os.path.join("../set-a","train","list.txt"),False,True)
    x=dt_train.x
    
    missing_rate=[[0 for col in range(41)] for row in range(48)]
    count=[[0 for col in range(41)] for row in range(48)]
    #3595*48
    for sample in x:
        for i in range(len(sample)):
            for j in range(len(sample[i])):
                if sample[i][j]!=0:
                    count[i][j]+=1
                
    missing_rate=1-np.array(count)/3595.0
    #for i in range(41):
    #    missing_rate[i]=1-float(count[i])/3595.0/48.0
    #missing_rate=1-np.array(missing_rate)
    np.savetxt("miss.txt",delimiter=',',X=missing_rate)
    
    line=''
    for i in range(len(missing_rate)):
        for j in range(5,len(missing_rate[i])):
                temp="["+str(i)+","+str(j-5)+","+str(missing_rate[i][j])+"]"+","
                line+=temp
    
    with open('miss_results', 'w') as f:
        f.write(line)
    
    print(line)
    return x,count,missing_rate


x,count,missing_rate=f()
"""
0
0.000834492
0.472323
0
0.0720155
0.987854
0.983924
0.983484
0.983455
0.98338
0.927434
0.998389
0.927051
0.463387
0.840398
0.683432
0.932209
0.929039
0.90474
0.0979543
0.924878
0.958977
0.929161
0.464331
0.846268
0.92941
0.578715
0.583838
0.578576
0.883159
0.883264
0.87788
0.926408
0.763485
0.9s58814
0.463329
0.626976
0.997699
0.989105
0.317901
0.93269
"""