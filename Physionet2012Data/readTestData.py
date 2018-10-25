#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 27 22:10:03 2018

@author: lyh
"""

import os
import random
class ReadPhysionetData():
    # first read all dataset
    # before call, determine wheher shuffle
    # produce next batch
    def __init__(self, dataPath, labelPath, maxLength,isNormal,isSlicing,sliceGap=60):
        print("data path: "+labelPath)
        labelFile = open(labelPath)
        fileNames=[]
        labels=[]
        #dataset: filenames,labels
        line_num = 0 
        for line in  labelFile.readlines():
        # rstrip() remove spaces in right end
            if line_num!=0:
                words = line.strip().split(',') 
                if os.path.isfile(os.path.join(dataPath, words[0]+".txt")):
                    fileNames.append(words[0]+".txt" )
                    if words[-1]=="0":
                        labels.append([1,0])
                    if words[-1]=="1":
                        labels.append([0,1])
            line_num=line_num+1
        self.dataPath = dataPath
        self.fileNames = fileNames
        labelFile.close()
        dic={'time':-1,'Age':0,'Gender':1,'Height':2,'ICUType':3,'Weight':4,'Albumin':5,\
             'ALP':6,'ALT':7,'AST':8,'Bilirubin':9,'BUN':10,'Cholesterol':11,'Creatinine':12,\
             'DiasABP':13,'FiO2':14,'GCS':15,'Glucose':16,'HCO3':17,'HCT':18,'HR':19,\
             'K':20,'Lactate':21,'Mg':22,'MAP':23,'MechVent':24,'Na':25,'NIDiasABP':26,\
             'NIMAP':27,'NISysABP':28,'PaCO2':29,'PaO2':30,'pH':31,'Platelets':32,'RespRate':33,\
             'SaO2':34,'SysABP':35,'Temp':36,'TroponinI':37,'TroponinT':38,'Urine':39,'WBC':40}
    
        self.dic=dic
        mean=[0.0]*(len(dic)-1)
        meancount=[0]*(len(dic)-1)
        self.std=[0.0]*(len(dic)-1)
        self.mean=[0.0]*(len(dic)-1)
        x=[]
        times=[]
        non_in_dic_count=0
        # times: totalFilesLength*steps
        # x: totalFilesLength*steps*feature_length
        for fileName in fileNames:
            f=open(os.path.join(self.dataPath, fileName))
            count=0
            age=gender=height=icutype=weight=-1
            lastTime=0
            totalData=[]
            t_times=[]
            for line in f.readlines():
                if count > 1:
                    words=line.strip().split(",")
                    timestamp=words[0]
                    feature=words[1]
                    value=words[2]
                    
                    # 0 is missing value,orignl gender is 0/1 ,after preprocessing
                    # gender is 0/1/2(missing,male,female)
                    if timestamp == "00:00":
                        if feature=='Age':
                            age="0" if value=="-1" else value
                            #calcuate mean
                            if age !="0":
                                mean[self.dic[feature]]+=float(age)
                                meancount[self.dic[feature]]+=1
                        if feature=='Gender':
                            if value=="-1":
                                gender="0"
                            if value=="0":
                                gender="1"
                            if value=="1":
                                gender="2"
                            #calcuate mean
                            if gender !="0":
                                mean[self.dic[feature]]+=float(gender)
                                meancount[self.dic[feature]]+=1
                        if feature=='Height':
                            height="0" if value=="-1" else value
                            #calcuate mean
                            if height !="0":
                                mean[self.dic[feature]]+=float(height)
                                meancount[self.dic[feature]]+=1
                        if feature == 'ICUType':
                            icutype="0" if value=="-1" else value
                            #calcuate mean
                            if icutype !="0":
                                mean[self.dic[feature]]+=float(icutype)
                                meancount[self.dic[feature]]+=1
                        if feature=='Weight':
                            weight="0" if value=="-1" else value
                            #calcuate mean
                            if weight !="0":
                                mean[self.dic[feature]]+=float(weight)
                                meancount[self.dic[feature]]+=1
                    else:
                        if timestamp!=lastTime:
                            data=[0.0]*(len(dic)-1)
                            hourandminute=timestamp.split(":")
                            t_times.append(float(hourandminute[0])*60+float(hourandminute[1]))
                            data[0]=float(age)
                            data[1]=float(gender)
                            data[2]=float(height)
                            data[3]=float(icutype)
                            data[4]=float(weight)
                            
                            data[self.dic[feature]]=float(value)
                            mean[self.dic[feature]]+=float(value)
                            meancount[self.dic[feature]]+=1
                            
                            totalData.append(data)
                        else:
                            
                            totalData[-1][self.dic[feature]]=float(value)
                            mean[self.dic[feature]]+=float(value)
                            meancount[self.dic[feature]]+=1
                            
                            
                    lastTime=timestamp      
                count+=1
                #if len(totalData)==24:
                #    break;
            x.append(totalData)
            times.append(t_times)
            f.close()
        #print(len(x))
        #for i in range(len(x)):
            #print(fileNames[i]+"'steps is :"+ str(len(x[i]))+" lable is : "+str(labels[i]))
        #print( x[0][0])
        self.x=x
        self.y=labels
        self.times=times
        
        self.timeslicing(isSlicing,sliceGap)
        
        
        
        meanFile=open(os.path.join("./", "meanAndstd"))
        linecount=0
        for line in meanFile.readlines():
            words=line.split(",")
            mean[linecount]=float(words[0])
            self.mean[linecount]=float(words[0])
            self.std[linecount]=float(words[1])
            meancount[linecount]=float(words[2])
            linecount+=1
        meanFile.close()
        # normalization
        m=[] # mask 0/1
        
        for onefile in self.x:
            one_m=[]
            for oneclass in onefile:
                t_m=[0]*len(oneclass)
                for j in range(len(oneclass)):
                    if oneclass[j] !=0:
                        t_m[j]=1
                one_m.append(t_m)
            m.append(one_m)
            
        #second update x
        self.isNormal=isNormal
        self.normalization(isNormal) 
                        
        x_lengths=[] #
        deltaPre=[] #time difference 
        lastvalues=[] # if missing, last values
        deltaSub=[]
        subvalues=[]
        for h in range(len(self.x)):
            # oneFile: steps*value_number
            oneFile=self.x[h]
            one_time=self.times[h]
            x_lengths.append(len(oneFile))
            
            one_deltaPre=[]
            one_lastvalues=[]
            
            one_deltaSub=[]
            one_subvalues=[]
            
            one_m=m[h]
            for i in range(len(oneFile)):
                t_deltaPre=[0.0]*len(oneFile[i])
                t_lastvalue=[0.0]*len(oneFile[i])
                one_deltaPre.append(t_deltaPre)
                one_lastvalues.append(t_lastvalue)
                
                if i==0:
                    for j in range(len(oneFile[i])):
                        one_lastvalues[i][j]=0.0 if one_m[i][j]==0 else oneFile[i][j]
                    continue
                for j in range(len(oneFile[i])):
                    if one_m[i-1][j]==1:
                        one_deltaPre[i][j]=one_time[i]-one_time[i-1]
                    if one_m[i-1][j]==0:
                        one_deltaPre[i][j]=one_time[i]-one_time[i-1]+one_deltaPre[i-1][j]
                        
                    if one_m[i][j]==1:
                        one_lastvalues[i][j]=oneFile[i][j]
                    if one_m[i][j]==0:
                        one_lastvalues[i][j]=one_lastvalues[i-1][j]
        
            for i in range(len(oneFile)):
                t_deltaSub=[0.0]*len(oneFile[i])
                t_subvalue=[0.0]*len(oneFile[i])
                one_deltaSub.append(t_deltaSub)
                one_subvalues.append(t_subvalue)
            #construct array 
            for i in range(len(oneFile)-1,-1,-1):    
                if i==len(oneFile)-1:
                    for j in range(len(oneFile[i])):
                        one_subvalues[i][j]=0.0 if one_m[i][j]==0 else oneFile[i][j]
                    continue
                for j in range(len(oneFile[i])):
                    if one_m[i+1][j]==1:
                        one_deltaSub[i][j]=one_time[i+1]-one_time[i]
                    if one_m[i+1][j]==0:
                        one_deltaSub[i][j]=one_time[i+1]-one_time[i]+one_deltaSub[i+1][j]
                        
                    if one_m[i][j]==1:
                        one_subvalues[i][j]=oneFile[i][j]
                    if one_m[i][j]==0:
                        one_subvalues[i][j]=one_subvalues[i+1][j]   
                
            
            #m.append(one_m)
            deltaPre.append(one_deltaPre)
            lastvalues.append(one_lastvalues)
            deltaSub.append(one_deltaSub)
            subvalues.append(one_subvalues)
        self.m=m
        self.deltaPre=deltaPre
        self.lastvalues=lastvalues
        self.deltaSub=deltaSub
        self.subvalues=subvalues
        self.x_lengths=x_lengths
        self.maxLength=max(x_lengths)
        print("max_length is : "+str(self.maxLength))

    def normalization(self,isNormal):
        if not isNormal:
            return
        for onefile in self.x:
            for oneclass in onefile:
                for j in range(len(oneclass)):
                    if oneclass[j] !=0:
                        if self.std[j]==0:
                            oneclass[j]=0.0
                        else:
                            oneclass[j]=1.0/self.std[j]*(oneclass[j]-self.mean[j])
    def timeslicing(self,isSlicing,sliceGap):
        #slicing x, make time gap be 30min, get the average of 30min
        if not isSlicing:
            return
        else:
            newx=[]
            newtimes=[]
            for i in range(len(self.times)):
                nowx=self.x[i]
                nowtime=self.times[i]
                lasttime=0
                newnowx=[]
                newnowtime=[]
                count=[0.0]*(len(self.dic)-1)
                tempx=[0.0]*(len(self.dic)-1)
                #newnowx.append(tempx)
                #newnowtime.append(lasttime)
                nowtime.append(48*60+2)
                for j in range(len(nowtime)):
                    if nowtime[j]<=lasttime+sliceGap:
                        for k in range(0,len(self.dic)-1):
                            tempx[k]+=nowx[j][k]
                            if nowx[j][k]!=0:
                                count[k]+=1.0
                    else:
                        for k in range(0,len(self.dic)-1):
                            if count[k]==0:
                                count[k]=1.0
                            tempx[k]=tempx[k]/count[k]
                        while nowtime[j]>lasttime+sliceGap:
                            newnowx.append(tempx)
                            newnowtime.append(lasttime)
                            lasttime+=sliceGap
                            count=[0.0]*(len(self.dic)-1)
                            tempx=[0.0]*(len(self.dic)-1)
                        # j may be len(nowx), we add one point into nowtime before
                        if j<len(nowx):
                            for k in range(0,len(self.dic)-1):
                                tempx[k]+=nowx[j][k]
                                if nowx[j][k]!=0:
                                    count[k]+=1.0
                
                for j in range(len(newnowtime)):
                    if newnowx[j][0]==0:
                        newnowx[j][0]=nowx[0][0]
                    if newnowx[j][1]==0:
                        newnowx[j][1]=nowx[0][1]
                    if newnowx[j][2]==0:
                        newnowx[j][2]=nowx[0][2]
                    if newnowx[j][3]==0:
                        newnowx[j][3]=nowx[0][3]
                    if newnowx[j][4]==0:
                        newnowx[j][4]=nowx[0][4]
                            
                nowtime.pop(-1)
                newx.append(newnowx)
                newtimes.append(newnowtime)
            self.x=newx
            self.times=newtimes
            
    
    
    def nextBatch(self):
        i=1
        while i*self.batchSize<=len(self.x):
            x=[]
            y=[]
            m=[]
            deltaPre=[]
            x_lengths=[]
            lastvalues=[]
            deltaSub=[]
            subvalues=[]
            imputed_deltapre=[]
            imputed_m=[]
            imputed_deltasub=[]
            mean=self.mean
            files=[]
            for j in range((i-1)*self.batchSize,i*self.batchSize):
                files.append(self.fileNames[j])
                x.append(self.x[j])
                y.append(self.y[j])
                m.append(self.m[j])
                deltaPre.append(self.deltaPre[j])
                deltaSub.append(self.deltaSub[j])
                #放的都是引用，下面添加0，则原始数据也加了0
                x_lengths.append(self.x_lengths[j])
                lastvalues.append(self.lastvalues[j])
                subvalues.append(self.subvalues[j])
                jj=j-(i-1)*self.batchSize
                #times.append(self.times[j])
                while len(x[jj])<self.maxLength:
                    t1=[0.0]*(len(self.dic)-1)
                    x[jj].append(t1)
                    #times[jj].append(0.0)
                    t2=[0]*(len(self.dic)-1)
                    m[jj].append(t2)
                    t3=[0.0]*(len(self.dic)-1)
                    deltaPre[jj].append(t3)
                    t4=[0.0]*(len(self.dic)-1)
                    lastvalues[jj].append(t4)
                    t5=[0.0]*(len(self.dic)-1)
                    deltaSub[jj].append(t5)
                    t6=[0.0]*(len(self.dic)-1)
                    subvalues[jj].append(t6)
            for j in range((i-1)*self.batchSize,i*self.batchSize):
                one_imputed_deltapre=[]
                one_imputed_deltasub=[]
                one_G_m=[]
                for h in range(0,self.x_lengths[j]):
                    if h==0:
                        one_f_time=[0.0]*(len(self.dic)-1)
                        one_imputed_deltapre.append(one_f_time)
                        try:
                            one_sub=[self.times[j][h+1]-self.times[j][h]]*(len(self.dic)-1)
                        except:
                            print("error: "+str(h)+" "+str(len(self.times[j]))+" "+self.fileNames[j])
                        one_imputed_deltasub.append(one_sub)
                        one_f_g_m=[1.0]*(len(self.dic)-1)
                        one_G_m.append(one_f_g_m)
                    elif h==self.x_lengths[j]-1:
                        one_f_time=[self.times[j][h]-self.times[j][h-1]]*(len(self.dic)-1)
                        one_imputed_deltapre.append(one_f_time)
                        one_sub=[0.0]*(len(self.dic)-1)
                        one_imputed_deltasub.append(one_sub)
                        one_f_g_m=[1.0]*(len(self.dic)-1)
                        one_G_m.append(one_f_g_m)
                    else:
                        one_f_time=[self.times[j][h]-self.times[j][h-1]]*(len(self.dic)-1)
                        one_imputed_deltapre.append(one_f_time)
                        one_sub=[self.times[j][h+1]-self.times[j][h]]*(len(self.dic)-1)
                        one_imputed_deltasub.append(one_sub)
                        one_f_g_m=[1.0]*(len(self.dic)-1)
                        one_G_m.append(one_f_g_m)
                while len(one_imputed_deltapre)<self.maxLength:
                    one_f_time=[0.0]*(len(self.dic)-1)
                    one_imputed_deltapre.append(one_f_time)
                    one_sub=[0.0]*(len(self.dic)-1)
                    one_imputed_deltasub.append(one_sub)
                    one_f_g_m=[0.0]*(len(self.dic)-1)
                    one_G_m.append(one_f_g_m)
                imputed_deltapre.append(one_imputed_deltapre)
                imputed_deltasub.append(one_imputed_deltasub)
                imputed_m.append(one_G_m)
                #重新设置times,times和delta类似，但times生成的时候m全是1,用于生成器G
            i+=1
            if self.isNormal:
                yield  x,y,[0.0]*(len(self.dic)-1),m,deltaPre,x_lengths,lastvalues,files,imputed_deltapre,imputed_m,deltaSub,subvalues,imputed_deltasub
            else:
                yield  x,y,mean,m,deltaPre,x_lengths,lastvalues,files,imputed_deltapre,imputed_m,deltaSub,subvalues,imputed_deltasub
            
    def shuffle(self,batchSize=32,isShuffle=False):
        self.batchSize=batchSize
        if isShuffle:
            c = list(zip(self.x,self.y,self.m,self.deltaPre,self.x_lengths,self.lastvalues,self.fileNames,self.times,self.deltaSub,self.subvalues))
            random.shuffle(c)
            self.x,self.y,self.m,self.deltaPre,self.x_lengths,self.lastvalues,self.fileNames,self.times,self.deltaSub,self.subvalues=zip(*c)

if __name__ == '__main__'     :
    
    dt=ReadPhysionetData("/home/yonghong/ImputationAndPredictionUsingGAN/set-a/test", \
                         "/home/yonghong/ImputationAndPredictionUsingGAN/set-a/test/list.txt",48,\
                         isNormal=False,isSlicing=True)
    dt.shuffle(128,False)
    batchCount=1
    X_lengths=dt.x_lengths
    print(sum(X_lengths)/len(X_lengths))
    for x,y,mean,m,delta,x_lengths,lastvalues,files,times,data_G_m in dt.nextBatch():
        print(batchCount)
        batchCount+=1
        if batchCount%100==0:
            print(files)
def f():
    print("readData")

#dt.shuffle(2,True)
#for x2,y2,mean,m2,delta2,x_lengths2,lastvalues2,files2 in dt.nextBatch():
#    print(files2)

#dt.shuffle(2,True)
#for x3,y3,mean,m3,delta3,x_lengths3,lastvalues3,files3 in dt.nextBatch():
#    print(files3)

