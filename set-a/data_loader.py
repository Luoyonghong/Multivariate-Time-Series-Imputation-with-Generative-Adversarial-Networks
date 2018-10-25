from __future__ import print_function
import torch.utils.data as data
import torch
import os

class MyDataset(data.Dataset):
    def __init__(self, dataPath, labelPath):
        labelFile = open(labelPath)
        dataset=[]
        #dataset: filenames,labels
        line_num = 0 
        for line in  labelFile.readlines():
        # rstrip() remove spaces in right end
            if line_num!=0:
                words = line.strip().split(',') 
                if os.path.isfile(os.path.join(dataPath, words[0]+".txt")):
                    dataset.append((words[0]+".txt", words[len(words)-1]))
            line_num=line_num+1
        self.dataPath = dataPath
        self.dataset = dataset
        dic={'time':0,'Age':1,'Gender':2,'Height':3,'ICUType':4,'Weight':5,'Albumin':6,\
             'ALP':7,'ALT':8,'AST':9,'Bilirubin':10,'BUN':11,'Cholesterol':12,'Creatinine':13,\
             'DiasABP':14,'FiO2':15,'GCS':16,'Glucose':17,'HCO3':18,'HCT':19,'HR':20,\
             'K':21,'Lactate':22,'Mg':23,'MAP':24,'MechVent':25,'Na':26,'NIDiasABP':27,\
             'NIMAP':28,'NISysABP':29,'PaCO2':30,'PaO2':31,'pH':32,'Platelets':33,'RespRate':34,\
             'SaO2':35,'SysABP':36,'Temp':37,'TropI':38,'TropT':39,'Urine':40,'WBC':41}
    
        self.dic=dic
        #self.classes = class_names
        #self.transform = transform
        #self.target_transform = target_transform
        #self.loader = loader


    def __getitem__(self, index):
        #print(self.dataset[index])
        fileName, label = self.dataset[index]
        f=open(os.path.join(self.dataPath, fileName))
        #read_csv is DataFrame, need to be transformed into ndarray
        count=0
        age=gender=height=icutype=weight=-1
        lastTime=0
        totalData=[]
        for line in f.readlines():
            if count > 1:
                words=line.split(",")
                timestamp=words[0]
                feature=words[1]
                value=words[2]
                
                # -1 is missing value
                if timestamp == "00:00":
                    if feature=='Age':
                        age=value
                    if feature=='Gender':
                        gender=value
                    if feature=='Height':
                        height=value
                    if feature == 'ICUType':
                        icutype=value
                    if feature=='Weight':
                        weight=value
                else:
                    if timestamp!=lastTime:
                        data=[-1.0]*42
                        hourandminute=timestamp.split(":")
                        data[0]=float(hourandminute[0])*60+float(hourandminute[1])
                        data[1]=float(age)
                        data[2]=float(gender)
                        data[3]=float(height)
                        data[4]=float(icutype)
                        data[5]=float(weight)
                        data[self.dic[feature]]=float(value)
                        totalData.append(data)
                    else:
                        totalData[len(totalData)-1][self.dic[feature]]=float(value)
                lastTime=timestamp      
            count+=1
            #if len(totalData)==24:
            #    break;
        #print(totalData)
        #print(label)
        return torch.FloatTensor(totalData), label, fileName

    def __len__(self):
        return len(self.dataset)

dataset = MyDataset("/home/lyh/Desktop/set-a/train/", "/home/lyh/Desktop/set-a/train/list.txt")
train_loader = torch.utils.data.DataLoader(\
    dataset, batch_size=2, shuffle=False, num_workers=0)
print(len(train_loader))
for dataset,label,filename in train_loader:
    print(filename)
    #print(dataset)
    print(label)

#for dataset in train_loader:
#    print(dataset)
    


