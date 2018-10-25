#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 22:39:25 2018

@author: lyh
"""
from __future__ import print_function
import sys
sys.path.append("..")
import argparse
import os
import tensorflow as tf
from Physionet2012Data import readData ,readTestData
import gru_d 
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument('--gpus', type=str, default = None)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--run-type', type=str, default='train')
    parser.add_argument('--data-path', type=str, default="/Users/luoyonghong/Documents/set-a/train/")
    parser.add_argument('--model-path', type=str, default=None)
    parser.add_argument('--result-path', type=str, default=None)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epoch', type=int, default=20)
    parser.add_argument('--n-inputs', type=int, default=41)
    parser.add_argument('--n-hidden-units', type=int, default=64)
    parser.add_argument('--n-classes', type=int, default=2)
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoint_physionet_no',
                        help='Directory name to save the checkpoints')
    parser.add_argument('--log-dir', type=str, default='logs_physionet_no',
                        help='Directory name to save training logs')
    parser.add_argument('--isNormal',type=bool,default=0)
    parser.add_argument('--isSlicing',type=int,default=0)
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
    
    dt_train=readData.ReadPhysionetData(os.path.join(args.data_path,"train"), os.path.join(args.data_path,"train","list.txt"),isNormal=args.isNormal,isSlicing=args.isSlicing)
    dt_test=readTestData.ReadPhysionetData(os.path.join(args.data_path,"test"), os.path.join(args.data_path,"test","list.txt"),dt_train.maxLength,isNormal=args.isNormal,isSlicing=args.isSlicing)
    
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        model = gru_d.grud(sess,
                    args=args,
                    dataset=dt_train,
                    )

        # build graph
        model.build()

        model.train()
        print(" [*] Training finished!")
        model.test(dt_test)
        # visualize learned generator
        #gan.visualize_results(args.epoch-1)
        print(" [*] Test finished!")