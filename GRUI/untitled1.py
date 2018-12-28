#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 09:23:18 2018

@author: yonghong
"""
import os 

def f():

    folders =  os.listdir("./")
    globalMin=0.0
    for folder in folders:
        if os.path.isdir(folder):
            secondPaths=os.listdir("./"+folder)
            nowMin=0.0
            for s in secondPaths:
                #print(s)
                if os.path.isfile(os.path.join("./",folder,s,"result")):
                    with open(os.path.join("./",folder,s,"result"),"r") as f:
                        temp=0.0 
                        for line in f:
                            #print(line)
                            a=float(line.split(",")[2])
                            #print(a)
                            if a>nowMin:
                                nowMin=a
                            if a>globalMin:
                                globalMin=a
                            if a>temp:
                                temp=a 
                        ggg=open(os.path.join(folder,s,str(temp)),"w")
                        ggg.close()
            r=open(os.path.join(folder,str(nowMin)),"w")
            r.close()
            if nowMin>globalMin:
                globalMin=nowMin
    d=open(os.path.join(str(globalMin)),"w")
    d.close()
    print(globalMin)

if __name__=="__main__":
    f()
            
        
