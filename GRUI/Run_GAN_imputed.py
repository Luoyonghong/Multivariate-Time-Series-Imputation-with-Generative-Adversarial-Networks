#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 10:47:41 2018

@author: yonghong
"""

from __future__ import print_function
import sys
sys.path.append("..")
import argparse
import os
import tensorflow as tf
from Physionet2012ImputedData import readImputed
import gru_delta_forGAN 
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument('--gpus', type=str, default = None)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--run-type', type=str, default='test')
    parser.add_argument('--data-path', type=str, default="../Gan_Imputation/imputation_train_results/WGAN_no_mask/")
    #输入填充之后的训练数据集的完整路径 Gan_Imputation/imputation_train_results/WGAN_no_mask/30_8_128_64_0.001_400_True_True_True_0.15_0.5
    parser.add_argument('--model-path', type=str, default=None)
    parser.add_argument('--result-path', type=str, default=None)
    parser.add_argument('--lr', type=float, default=0.01)
    #parser.add_argument('--epoch', type=int, default=20)
    parser.add_argument('--n-inputs', type=int, default=41)
    parser.add_argument('--n-hidden-units', type=int, default=64)
    parser.add_argument('--n-classes', type=int, default=2)
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoint_physionet_imputed',
                        help='Directory name to save the checkpoints')
    parser.add_argument('--log-dir', type=str, default='logs_physionet_imputed',
                        help='Directory name to save training logs')
    parser.add_argument('--isNormal',type=int,default=1)
    parser.add_argument('--isSlicing',type=int,default=1)
    #0 false 1 true
    parser.add_argument('--isBatch-normal',type=int,default=1)
    args = parser.parse_args()
    
    
    if args.isBatch_normal==0:
            args.isBatch_normal=False
    if args.isBatch_normal==1:
            args.isBatch_normal=True
    if args.isNormal==0:
            args.isNormal=False
    if args.isNormal==1:
            args.isNormal=True
    if args.isSlicing==0:
            args.isSlicing=False
    if args.isSlicing==1:
            args.isSlicing=True
            
    
    checkdir=args.checkpoint_dir
    logdir=args.log_dir
    base=args.data_path
    data_paths=["30_8_128_64_0.001_400_True_True_True_0.15_0.5"]
    max_auc = 0.0
    for d in data_paths:
        args.data_path=os.path.join(base,d)
        path_splits=args.data_path.split("/")
        if len(path_splits[-1])==0:
            datasetName=path_splits[-2]
        else:
            datasetName=path_splits[-1]
        args.checkpoint_dir=checkdir+"/"+datasetName
        args.log_dir=logdir+"/"+datasetName
        
        dt_train=readImputed.ReadImputedPhysionetData(args.data_path)
        dt_train.load()
        
        dt_test=readImputed.ReadImputedPhysionetData(args.data_path.replace("imputation_train_results","imputation_test_results"))
        dt_test.load()
          
        lrs=[0.004,0.003,0.005,0.006,0.007,0.008,0.009,0.01,0.012,0.015]
        #lrs = [0.0075,0.0085]
        for lr in lrs:
            args.lr=lr
            epoch=30
            args.epoch=epoch
            print("epoch: %2d"%(epoch))
            tf.reset_default_graph()
            config = tf.ConfigProto() 
            config.gpu_options.allow_growth = True 
            with tf.Session(config=config) as sess:
                model = gru_delta_forGAN.grui(sess,
                            args=args,
                            dataset=dt_train,
                            test_set = dt_test
                            )
            
                # build graph
                model.build()
            
                auc = model.train()
                if auc > max_auc:
                    max_auc = auc 
                
            print("")
        print("max auc is: " + str(max_auc))
        f2 = open("max_auc","w")
        f2.write(str(max_auc))
        f2.close()


