def main():
    
    import torch
    import torchvision
    import numpy as np
    import pandas as pd
    import torch.nn as nn
    import torch.nn.functional as F                 
    import torchvision.transforms.functional as TF 
    import torchvision.transforms as transforms
    import torchvision.models as models
    import os

    from torchvision import datasets, transforms
    from numpy import vstack
    from pandas import read_csv
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import f1_score
    from sklearn.metrics import roc_auc_score
    from sklearn.metrics import recall_score
    from torch.utils.data import Dataset
    from torch.utils.data import DataLoader
    from torch.utils.data import random_split
    from torch import Tensor
    from PIL import Image
    #from torchsummary import summary  
    from torchvision.datasets import ImageFolder
    from torchvision.transforms import ToTensor
    from torch.nn import Linear
    from torch.nn import ReLU
    from torch.nn import Sigmoid
    from torch.nn import Module
    from torch.optim import SGD
    from torch.optim import Adam
    from torch.nn import BCELoss
    from torch.nn.init import kaiming_uniform_
    from torch.nn.init import xavier_uniform_
    torch.cuda.empty_cache()
    transform = transforms.Compose([transforms.ToTensor()]) # Rescales all images to betwee 0 and 1
    #CUDA_LAUNCH_BLOCKING=1
    F1 = f1_score
    Acc = accuracy_score
    Sn = recall_score
    def Sp(actualls,predictions):
        tn =0
        n=0
        for i in range(len(actualls)):
            if((actualls[i]==0)and(predictions[i]==0)):
                tn =tn+1
            if(actualls[i]==0):
                n = n+1

        return (tn/n)

    def get_default_device():
        """Pick GPU if available, else CPU"""
        if torch.cuda.is_available():
            return torch.device('cuda')
        else:
            return torch.device('cpu')

    def to_device(data, device):
        """Move tensor(s) to chosen device"""
        if isinstance(data, (list,tuple)):
            return [to_device(x, device) for x in data]
        return data.to(device, non_blocking=True)

    class DeviceDataLoader():
        """Wrap a dataloader to move data to a device"""
        def __init__(self, dl, device):
            self.dl = dl
            self.device = device

        def __iter__(self):
            """Yield a batch of data after moving it to device"""
            for b in self.dl: 
                yield to_device(b, self.device)

        def __len__(self):
            """Number of batches"""
            return len(self.dl)
    class CSVDataset(Dataset):
        #To load and preprocess Clinical Data
        # load the dataset
        def __init__(self, path):
            # load the csv file as a dataframe
            df = read_csv(path, skiprows=1) # we skip the first row containing labels
            # store the inputs and outputs
            self.X = df.values[:, :-1] #split the last label from the rest of the data
            self.y = df.values[:, -1]
            # ensure input data is floats
            self.X = self.X.astype('float32')
            # label encode target and ensure the values are floats
            self.y = LabelEncoder().fit_transform(self.y)
            self.y = self.y.astype('float32')
            self.y = self.y.reshape((len(self.y), 1))
        # number of rows in the dataset
        def __len__(self):
            return len(self.X)

        # get a row at an index
        def __getitem__(self, idx):
            return [torch.tensor(self.X[idx], dtype = torch.float32),
                    torch.tensor(self.y[idx], dtype = torch.float32)]

        # get indexes for train and test rows
        def get_splits(self, n_test=0.33):
            # determine sizes
            test_size = round(n_test * len(self.X))
            train_size = len(self.X) - test_size
            # calculate the split
            return random_split(self, [train_size, test_size])
    def prepare_data(path):
        # load the dataset
        dataset = CSVDataset(path)
        dataset = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        #dataset = torch.tensor(dataset, dtype = torch.float32)
        # prepare data loaders
        return dataset
     
    class MLP(Module):
        # For Clinical Data
        # define model elements
        def __init__(self, n_inputs):
                        
            super(MLP, self).__init__()
            # input to first hidden layer
            self.hidden1 = Linear(n_inputs, 32)
            kaiming_uniform_(self.hidden1.weight, nonlinearity='relu')
            self.act1 = ReLU()
            # second hidden layer
            self.hidden2 = Linear(32, 128)
            kaiming_uniform_(self.hidden2.weight, nonlinearity='relu')
            self.act2 = ReLU()
            # third hidden layer 
            self.hidden3 = Linear(128, 512)
            kaiming_uniform_(self.hidden3.weight, nonlinearity='relu')
            self.act3 = ReLU()
            # fourth hidden layer
            self.hidden4 = Linear(512,1280)
            xavier_uniform_(self.hidden3.weight)
            self.act4 = ReLU()
            # outpute layer
            self.output = Linear(1280,1)
            self.act5 = Sigmoid()

        # forward propagate input
        def forward(self, X):
            
            # input to first hidden layer
            X = self.hidden1(X)
            X = self.act1(X)
             # second hidden layer
            X = self.hidden2(X)
            X = self.act2(X)
            # third hidden layer and output
            X = self.hidden3(X)
            X = self.act3(X)
            X = self.hidden4(X)
            X  = self.act4(X)
            X  =self.output(X)
            X = self.act5(X)
            return X
 
    
    device =get_default_device()
    #device  ='cpu'
    print(device)
    batch_size =32
    epochs = 10

    XRdatasettrain =  ImageFolder(r"./X-Ray Train", transform=transform)
    XRdatasettest  = ImageFolder(r"./X-Ray Test",transform =transform)
    XRTrain = DataLoader(XRdatasettrain, batch_size,shuffle =True)
    XRTest = DataLoader(XRdatasettest, batch_size,shuffle =True)
    XRTrain = DeviceDataLoader(XRTrain, device)
    XRTest = DeviceDataLoader(XRTest, device)
    CTdatasettrain = ImageFolder(r"./CT-Scan Train",transform=transform)
    CTdatasettest = ImageFolder(r"./CT-Scan Test",transform=transform)
    CTTrain = DataLoader(CTdatasettrain, batch_size,shuffle =True)
    CTTest = DataLoader(CTdatasettest, batch_size,shuffle =True)
    CTTrain = DeviceDataLoader(CTTrain, device)
    CTTest = DeviceDataLoader(CTTest, device)
    CLdatasettrain  = prepare_data(r"./TrainClinical.csv")
    CLdatasettest = prepare_data(r"./TestClinical.csv")
    CLTrain = DeviceDataLoader(CLdatasettrain, device)
    CLTest = DeviceDataLoader(CLdatasettest, device)

    EfficientB0 = models.efficientnet_b0(pretrained=False).to(device)
    EfficientB1 = models.efficientnet_b1(pretrained=False).to(device)
    Resnet50XR  = models.resnet50(pretrained=False).to(device)
    Resnet50CT  = models.resnet50(pretrained=False).to(device)
    CL1 = MLP(8).to(device)
    CL2 = MLP(8).to(device)
    
    #*********************************************************Processing*****************************************************************#
    EfficientB0.classifier = nn.Sequential(
                             nn.Dropout(p=0.2, inplace=True),
                             nn.Linear(1280,1),
                              )
    EfficientB1.classifier = nn.Sequential(
                         nn.Dropout(p=0.2, inplace=True),
                         nn.Linear(1280,1),
                          )
    Resnet50XR.fc  = nn.Sequential(
                      nn.Linear(2048,1280),
                      nn.LeakyReLU()
                        )
    Resnet50XR.classifier  = nn.Sequential(
                      
                      nn.Linear(1280,1),
          
                   )  
    
    Resnet50CT.fc  = nn.Sequential(
                      nn.Linear(2048,1280),
                      nn.LeakyReLU()   
                   )
    Resnet50CT.classifier  = nn.Sequential(
                      
                      nn.Linear(1280,1),
          
                   ) 
    EfficientB0 = to_device(EfficientB0,device)
    EfficientB1 = to_device(EfficientB1,device)
    Resnet50XR    = to_device(Resnet50XR,device)
    Resnet50CT    = to_device(Resnet50CT,device)                      

                                
    #*********************************************************Pre-Training****************************************************************#
    
    
    optim_1 = torch.optim.Adam(EfficientB0.parameters(),lr =0.001)
    criterion_1 = nn.BCEWithLogitsLoss()
    epochs =10
    for epoch in range(epochs):        
        for i,(X,y) in enumerate(XRTrain):
            f = X.shape[0]
            y = y.reshape((len(y),1))
            y = y.to(torch.float)
            X = EfficientB0(X)
            X = X.reshape((f,-1))
            X = X.to(torch.float)
            optim_1.zero_grad()
            loss_1 = criterion_1(X,y)
            loss_1.backward()
            optim_1.step()
        #print(epoch)
    #print("DONE")
    optim_2 = torch.optim.Adam(Resnet50XR.parameters(),lr =0.001)
    criterion_2 = nn.BCEWithLogitsLoss()
    
    for epoch in range(epochs):        
        for i,(X,y) in enumerate(XRTrain):
            f = X.shape[0]
            y = y.reshape((len(y),1))
            y = y.to(torch.float)
            X = Resnet50XR(X)
            
            X = Resnet50XR.classifier(X)
            #print(X.shape)
            X = X.reshape((f,-1))
            X = X.to(torch.float)
            optim_2.zero_grad()
            loss_2 = criterion_2(X,y)
            loss_2.backward()
            optim_2.step()
            
    #print("DONE")
    optim_3 = torch.optim.Adam(EfficientB1.parameters(),lr =0.001)
    criterion_3 = nn.BCEWithLogitsLoss()
    for epoch in range(epochs):        
        for i,(X,y) in enumerate(CTTrain):
            f = X.shape[0]
            y  = y.reshape((len(y),1))
            y  = y.to(torch.float)
            X = EfficientB1(X)
            X  = X.reshape((f,-1))
            X = X.to(torch.float)
            optim_3.zero_grad()
            loss_3 = criterion_3(X,y)
            loss_3.backward()
            optim_3.step()
    #print("DONE")
    optim_4 = torch.optim.Adam(Resnet50CT.parameters(),lr =0.001)
    criterion_4 = nn.BCEWithLogitsLoss()
    
    for epoch in range(epochs):        
        for i,(X,y) in enumerate(CTTrain):
            f = X.shape[0]
            y  = y.reshape((len(y),1))
            y  = y.to(torch.float)
            X = Resnet50CT(X)
            #print(X.shape)
            X = Resnet50CT.classifier(X)
            #print(X.shape)
            X  = X.reshape((f,-1))
            X = X.to(torch.float)
            optim_4.zero_grad()
            loss_4 = criterion_4(X,y)
            loss_4.backward()
            optim_4.step()
    #print("DONE")        
    optim_5 = torch.optim.Adam(CL1.parameters(),lr =0.001)
    criterion_5 = nn.BCEWithLogitsLoss()
    
    for epoch in range(epochs+5):        
        for i,(X,y) in enumerate(CLTrain):
            f = X.shape[0]
            y  = y.reshape((len(y),1))
            y  = y.to(torch.float)
            X = CL1(X)
            X  = X.reshape((f,-1))
            X = X.to(torch.float)
            optim_5.zero_grad()
            loss_5 = criterion_5(X,y)
            loss_5.backward()
            optim_5.step()
    #print("DONE")
    optim_6 = torch.optim.Adam(CL2.parameters(),lr =0.001)
    criterion_6 = nn.BCEWithLogitsLoss()
    
    for epoch in range(epochs+5):        
        for i,(X,y) in enumerate(CLTrain):
            f = X.shape[0]
            y  = y.reshape((len(y),1))
            y  = y.to(torch.float)
            X = CL2(X)
            X  = X.reshape((f,-1))
            X = X.to(torch.float)
            optim_6.zero_grad()
            loss_6 = criterion_6(X,y)
            loss_6.backward()
            optim_6.step()
           
    print("Pre Training DONE")
    
    #*********************************************************Setting****************************************************************#
    EfficientB0 = nn.Sequential(*list(EfficientB0.children())[:-1])
    EfficientB1 = nn.Sequential(*list(EfficientB1.children())[:-1])
    
    EfficientB0 = to_device(EfficientB0,device)
    EfficientB1 = to_device(EfficientB1,device)
    Resnet50XR    = to_device(Resnet50XR,device)
    Resnet50CT    = to_device(Resnet50CT,device)
    
    SCL = nn.Sequential(*list(CL1.children())[:-2])  
    TCL = nn.Sequential(*list(CL2.children())[:-2])
    SCL = to_device(SCL,device)
    TCL = to_device(TCL,device)
    
    #*********************************************************Parameters****************************************************************#
    
    Param_G    = []
    Param_D    = []
    Param_cxr  = []
    Param_cct  = []
    Param_ccl  = []
    Param_txr  = []
    Param_tct  = []
    Param_tcl  = []    
    Param_sxr  = []
    Param_sct  = []
    Param_scl  = []
    
    
    
    
    
    
   
    
    
    
    Criterion_cxr  = nn.BCEWithLogitsLoss()
    Criterion_cct  = nn.BCEWithLogitsLoss()
    Criterion_ccl  = nn.BCEWithLogitsLoss()
    Criterion_txr  = nn.BCEWithLogitsLoss()
    Criterion_tct  = nn.BCEWithLogitsLoss()
    Criterion_tcl  = nn.BCEWithLogitsLoss()
    Criterion_g    = nn.BCEWithLogitsLoss()
    Criterion_da   = nn.BCEWithLogitsLoss()
    Criterion_df   = nn.BCEWithLogitsLoss()
    
    
    
    
    
    #*********************************************************XR***********************************************************************#
    #print(Resnet50XR)
    
    TaskXR = nn.Sequential(
                #nn.BatchNorm2d(1280),
                nn.Linear(1280,960),
                nn.LeakyReLU(),
                nn.Linear(960,640),
                nn.LeakyReLU(),                           
                nn.Linear(640,1),
                
                       )   
    to_device(TaskXR,device)
    def TaskembedXR(X):
        f = X.shape[0]
        X = EfficientB0(X)
        X  = X.reshape((f,-1)) 
        return X
    
    
    Param_txr.append(TaskXR[0].weight)
    Param_txr.append(TaskXR[2].weight)
    Param_txr.append(TaskXR[4].weight)
    Optim_txr  = torch.optim.Adam(Param_txr, lr=0.001)
    
    def SharedembedXR(X):
        
        f = X.shape[0]
        X = Resnet50XR(X)
        X  = X.reshape((f,-1))
        return X
    
    
    ClassifierheadXR = nn.Sequential(
            nn.Dropout(p=0.2),                   
            nn.Linear(1920,320),
            nn.LeakyReLU(),            
            nn.Linear(320,160),
            nn.LeakyReLU(),
            nn.Linear(160,1)


        )
    to_device(ClassifierheadXR,device)
    
    Param_cxr.append(ClassifierheadXR[1].weight)
    Param_cxr.append(ClassifierheadXR[3].weight)
    Param_cxr.append(ClassifierheadXR[5].weight)   
    Optim_cxr  = torch.optim.Adam(Param_cxr, lr=0.001)
    
    
    #*********************************************************CT***********************************************************************#

    
    TaskCT = nn.Sequential(
                #nn.BatchNorm2d(1280),    
                nn.Linear(1280,960),
                nn.LeakyReLU(),
                nn.Linear(960,640),
                nn.LeakyReLU(),
                       
                nn.Linear(640,1),
                       )
    to_device(TaskCT,device)
    
    def TaskembedCT(X):
        f = X.shape[0]
        X = EfficientB1(X)
        X  = X.reshape((f,-1))        
        return X
    
    Param_tct.append(TaskCT[0].weight)
    Param_tct.append(TaskCT[2].weight)
    Param_tct.append(TaskCT[4].weight)    
    Optim_tct  = torch.optim.Adam(Param_tct, lr=0.001)
    def SharedembedCT(X):
        f = X.shape[0]
        X = Resnet50CT(X)
        X  = X.reshape((f,-1))        
        return X
    
    
    ClassifierheadCT = nn.Sequential(
            nn.Dropout(p=0.2),                   
            nn.Linear(1920,320),
            nn.LeakyReLU(),            
            nn.Linear(320,160),
            nn.LeakyReLU(),
            nn.Linear(160,1)


        )
    to_device(ClassifierheadCT,device)
    
    Param_cct.append(ClassifierheadCT[1].weight)
    Param_cct.append(ClassifierheadCT[3].weight)
    Param_cct.append(ClassifierheadCT[5].weight)
    Optim_cct  = torch.optim.Adam(Param_cct, lr=0.001)
    
    #*********************************************************CL***********************************************************************#
    def TaskembedCL(X):
        f = TCL(X)
        return f
    def SharedembedCL(X):
        f = SCL(X)
        return f
    TaskCL = nn.Sequential(                      
            nn.Linear(1280,320),
            nn.LeakyReLU(),            
            nn.Linear(320,160),
            nn.LeakyReLU(),
            nn.Linear(160,1)
                       )
    to_device(TaskCL,device)
    Param_tcl = TaskCL.parameters()
    Optim_tcl  = torch.optim.Adam(Param_tcl, lr=0.001)    
    ClassifierheadCL = nn.Sequential(
            nn.Dropout(p=0.2),                   
            nn.Linear(1440,320),
            nn.LeakyReLU(),            
            nn.Linear(320,160),
            nn.LeakyReLU(),
            nn.Linear(160,1)


        )
    Param_ccl.append(ClassifierheadCT[1].weight)
    Param_ccl.append(ClassifierheadCT[3].weight)
    Param_ccl.append(ClassifierheadCT[5].weight)
    Optim_ccl  = torch.optim.Adam(Param_ccl, lr=0.001)
    to_device(ClassifierheadCL,device)
             
        
   #*********************************************************G***********************************************************************#     

    Generator = nn.Sequential(
        nn.Linear(1280,320),
        nn.LeakyReLU(),
        nn.Linear(320,160),
        nn.LeakyReLU(),
        nn.Linear(160,320),
        nn.LeakyReLU(),
        nn.Linear(320,1280),
        nn.LeakyReLU()
    )
    
    Param_G.append(Generator[0].weight)
    Param_G.append(Generator[2].weight)
    Param_G.append(Generator[4].weight)
    Param_G.append(Generator[6].weight)
    
    Generator = to_device(Generator,device)
    Optim_G    = torch.optim.Adam(Param_G, lr=0.001)
    Optim_Gf   = torch.optim.Adam(Param_G, lr=0.00001)
    
    def G_Loss(GenEmbed,InvLabel):
        T_L = Criterion_g(GenEmbed, InvLabel)
        
        return T_L
   #*********************************************************D***********************************************************************#
    Discriminator =nn.Sequential(
        nn.Linear(1280,320),
        nn.LeakyReLU(),
        nn.Linear(320,160),
        nn.LeakyReLU(),
        nn.Linear(160,3),
        

    )
    
    Param_D.append(Discriminator[0].weight)
    Param_D.append(Discriminator[2].weight)
    Param_D.append(Discriminator[4].weight)
    Optim_D    = torch.optim.Adam(Param_D, lr=0.001)
    Optim_Df   = torch.optim.Adam(Param_D, lr=0.00001)
    
    Discriminator = to_device(Discriminator,device)

    

    def D_Loss(ActEmbed,GenEmbed,Label,InvLabel):
               
        A_L = Criterion_da(ActEmbed, Label)
        F_L = Criterion_df(GenEmbed, InvLabel)
        T_L = F_L +A_L
        return T_L
    
       
 #*********************************************************Pre-training1******************************************************************#
    
    epochs =10
    for epoch in range(epochs):        
        for i,(X,y) in enumerate(XRTrain):
            f = X.shape[0]
            y  = y.reshape((len(y),1))
            y  = y.to(torch.float)
            X = EfficientB0(X)
            X  = X.reshape((f,-1))
            X = TaskXR(X)
            X  = X.reshape((f,-1))
            X = X.to(torch.float)
            Optim_txr.zero_grad()
            loss = Criterion_txr(X,y)
            loss.backward()
            Optim_txr.step()
    
    
    #print("DONE")
    for epoch in range(epochs):        
        for i,(X,y) in enumerate(CTTrain):
            f = X.shape[0]
            y  = y.reshape((len(y),1))
            y  = y.to(torch.float)
            X = EfficientB1(X)
            X  = X.reshape((f,-1))           
            X = TaskCT(X)
            X  = X.reshape((f,-1))
            X = X.to(torch.float)
            Optim_tct.zero_grad()
            loss = Criterion_tct(X,y)
            loss.backward()
            Optim_tct.step()
    
    #print("DONE")
    for epoch in range(epochs+5):        
        for i,(X,y) in enumerate(CLTrain):
            f = X.shape[0]
            y  = y.reshape((len(y),1))
            y  = y.to(torch.float)
            X  = TCL(X)
            X  = X.reshape((f,-1))
            X = TaskCL(X)
            X  = X.reshape((f,-1))
            X = X.to(torch.float)
            Optim_tcl.zero_grad()
            loss = Criterion_tcl(X,y)
            loss.backward()
            Optim_tcl.step()
    print("Pre Training 1 Done")  
 #***********************************************************Setting*********************************************************************#
    TaskXR = nn.Sequential(*list(TaskXR.children())[:-2])
    TaskCT = nn.Sequential(*list(TaskCT.children())[:-2])
    TaskCL = nn.Sequential(*list(TaskCL.children())[:-2])
    TaskXR = to_device(TaskXR,device)
    TaskCT = to_device(TaskCT,device)
    TaskCL = to_device(TaskCL,device)
    
 #*********************************************************Adversarial-Learning**********************************************************#     
    epochs = 10
    for epoch in range(epochs):
        
        for i,(X,y) in enumerate(XRTrain):
            #print(X.shape)
            f = X.shape[0]
            y  = y.reshape((len(y),1))
            y  = y.to(torch.float)
            Optim_D.zero_grad()
            #S = SharedembedXR(X)
            S = Resnet50XR(X)
            
            F = Generator(S)
            R_O = Discriminator(S)
            F_O = Discriminator(F)
            Label  = torch.zeros(f,3).to(device)
            InvLabel = torch.zeros(f,3).to(device)
            Label[:0] =1
            InvLabel[:1] = 1
            InvLabel[:2] =1
            dloss = D_Loss(R_O,F_O,Label,InvLabel) 
            #dloss.requires_grad = True
            dloss.backward()
            Optim_D.step()      
        
        
        for i,(X,y) in enumerate(CTTrain):
            f = X.shape[0]
            y  = y.reshape((len(y),1))
            y  = y.to(torch.float)
            Optim_D.zero_grad()
            S = Resnet50CT(X)
            F = Generator(S)
            R_O = Discriminator(S)
            F_O = Discriminator(F)
            
            Label  = torch.zeros(f,3).to(device)
            InvLabel = torch.zeros(f,3).to(device)
            Label[:1] =1
            InvLabel[:0] = 1
            InvLabel[:2] =1
            dloss = D_Loss(R_O,F_O,Label,InvLabel)  
            #dloss.requires_grad = True
            dloss.backward()                
            Optim_D.step()
         
        for i,(X,y) in enumerate(CLTrain):
            f = X.shape[0]
            y  = y.reshape((len(y),1))
            y  = y.to(torch.float)
            Optim_D.zero_grad()
            S = SharedembedCL(X)
            F = Generator(S)
            R_O = Discriminator(S)
            F_O = Discriminator(F)            
            Label  = torch.zeros(f,3).to(device)
            InvLabel = torch.zeros(f,3).to(device)
            Label[:2] =1
            InvLabel[:0] = 1
            InvLabel[:1] =1
            dloss = D_Loss(R_O,F_O,Label,InvLabel)  
            #dloss.requires_grad = True
            dloss.backward()                
            Optim_D.step()
    #print("DONE")           
    for epoch in range(5):
        
        for i,(X,y) in enumerate(XRTrain):
            f = X.shape[0]
            y  = y.reshape((len(y),1))
            y  = y.to(torch.float)
            Optim_Df.zero_grad()
            #S = SharedembedXR(X)
            S = Resnet50XR(X)
            F = Generator(S)
            R_O = Discriminator(S)
            F_O = Discriminator(F)
            Label  = torch.zeros(f,3).to(device)
            InvLabel = torch.zeros(f,3).to(device)
            Label[:0] =1
            InvLabel[:1] = 1
            InvLabel[:2] =1
            dloss = D_Loss(R_O,F_O,Label,InvLabel) 
            #dloss.requires_grad = True
            dloss.backward()
            Optim_Df.step()      
        

        for i,(X,y) in enumerate(CTTrain):
            f = X.shape[0]
            y  = y.reshape((len(y),1))
            y  = y.to(torch.float)
            Optim_Df.zero_grad()
            #S = SharedembedCT(X)
            S = Resnet50CT(X)
            F = Generator(S)
            R_O = Discriminator(S)
            F_O = Discriminator(F)
            
            Label  = torch.zeros(f,3).to(device)
            InvLabel = torch.zeros(f,3).to(device)
            Label[:1] =1
            InvLabel[:0] = 1
            InvLabel[:2] =1
            dloss = D_Loss(R_O,F_O,Label,InvLabel) 
            #dloss.requires_grad = True
            dloss.backward()                
            Optim_Df.step()
         
        for i,(X,y) in enumerate(CLTrain):
            f = X.shape[0]
            y  = y.reshape((len(y),1))
            y  = y.to(torch.float)
            Optim_Df.zero_grad()
            S = SharedembedCL(X)
            F = Generator(S)
            R_O = Discriminator(S)
            F_O = Discriminator(F)            
            Label  = torch.zeros(f,3).to(device)
            InvLabel = torch.zeros(f,3).to(device)
            Label[:2] =1
            InvLabel[:0] = 1
            InvLabel[:1] =1
            dloss = D_Loss(R_O,F_O,Label,InvLabel) 
            #dloss.requires_grad = True
            dloss.backward()                
            Optim_Df.step()    
    
    #print("DONE")                   
    for epoch in range(epochs):
        
        for i,(X,y) in enumerate(XRTrain):
            
            f = X.shape[0]
            y  = y.reshape((len(y),1))
            y  = y.to(torch.float)
            Optim_G.zero_grad()
            S = SharedembedXR(X)
            F = Generator(S)
            R_O = Discriminator(S)
            F_O = Discriminator(F)            
            InvLabel = torch.zeros(f,3).to(device)
            InvLabel[:1] = 1
            InvLabel[:2] =1
            gloss = G_Loss(F_O,InvLabel)    
            #gloss.requires_grad =True 
            gloss.backward()
            Optim_G.step()

        for i,(X,y) in enumerate(CTTrain):

            f = X.shape[0]
            y  = y.reshape((len(y),1))
            y  = y.to(torch.float)
            Optim_G.zero_grad()
            S = SharedembedCT(X)
            F = Generator(S)
            R_O = Discriminator(S)
            F_O = Discriminator(F)
            InvLabel = torch.zeros(f,3).to(device)
            InvLabel[:0] = 1
            InvLabel[:2] =1
            gloss = G_Loss(F_O,InvLabel)        
            #gloss.requires_grad =True 
            gloss.backward()
            Optim_G.step()
         
        for i,(X,y) in enumerate(CLTrain):

            f = X.shape[0]
            y  = y.reshape((len(y),1))
            y  = y.to(torch.float)
            Optim_G.zero_grad()
            S = SharedembedCL(X)
            F = Generator(S)
            R_O = Discriminator(S)
            F_O = Discriminator(F)
            InvLabel = torch.zeros(f,3).to(device)
            InvLabel[:0] = 1
            InvLabel[:1] =1
            gloss = G_Loss(F_O,InvLabel)  
            #gloss.requires_grad =True 
            gloss.backward()
            Optim_G.step()
    #print("DONE")            
    for epoch in range(5):
        
        for i,(X,y) in enumerate(XRTrain):
            
            f = X.shape[0]
            y  = y.reshape((len(y),1))
            y  = y.to(torch.float)
            Optim_Gf.zero_grad()
            S = SharedembedXR(X)
            F = Generator(S)
            R_O = Discriminator(S)
            F_O = Discriminator(F)            
            InvLabel = torch.zeros(f,3).to(device)
            InvLabel[:1] = 1
            InvLabel[:2] =1
            gloss = G_Loss(F_O,InvLabel)
            #gloss.requires_grad =True 
            gloss.backward()
            Optim_Gf.step()

        for i,(X,y) in enumerate(CTTrain):

            f = X.shape[0]
            y  = y.reshape((len(y),1))
            y  = y.to(torch.float)
            Optim_Gf.zero_grad()
            S = SharedembedCT(X)
            F = Generator(S)
            R_O = Discriminator(S)
            F_O = Discriminator(F)
            InvLabel = torch.zeros(f,3).to(device)
            InvLabel[:0] = 1
            InvLabel[:2] =1
            gloss = G_Loss(F_O,InvLabel) 
            #gloss.requires_grad =True 
            gloss.backward()
            Optim_Gf.step()
         
        for i,(X,y) in enumerate(CLTrain):

            f = X.shape[0]
            y  = y.reshape((len(y),1))
            y  = y.to(torch.float)
            Optim_Gf.zero_grad()
            S = SharedembedCL(X)
            F = Generator(S)
            R_O = Discriminator(S)
            F_O = Discriminator(F)
            InvLabel = torch.zeros(f,3).to(device)
            InvLabel[:0] = 1
            InvLabel[:1] =1
            gloss = G_Loss(F_O,InvLabel)  
            #gloss.requires_grad =True 
            gloss.backward()
            Optim_Gf.step()
    print("Adversarial Learning Done")       
  #*********************************************************Training**********************************************************# 
           
                
    epochs =10
                
    for epoch in range(epochs):
        XRLoss =0
        XRActuals = list()
        XRPredictions = list()
        CTLoss =0
        CTActuals = list()
        CTPredictions = list()
        CLLoss =0
        CLActuals = list()
        CLPredictions = list()
        for i,(X,y) in enumerate(XRTrain):

            f  = X.shape[0]
            y  = y.reshape((len(y),1))
            y  = y.to(torch.float)
            T  = TaskembedXR(X)
            S  = SharedembedXR(X)
            F  = Generator(S)
            T  = TaskXR(T)
            C  = torch.cat((T,F),1)
            Output = ClassifierheadXR(C)
            Output = Output.to(torch.float)
            Optim_cxr.zero_grad()
            lossxr   = Criterion_cxr(Output, y)
            lossxr.backward()
            XRLoss += lossxr
            Optim_cxr.step()
            Output = torch.round(torch.sigmoid(Output))
            Output =  Output.cpu().detach().numpy()
            y      =  y.cpu().detach().numpy()      
            XRActuals.append(y)
            XRPredictions.append(Output)

        Epoch_Loss = XRLoss/(len(XRTrain))
        XRPredictions, XRActuals = vstack(XRPredictions), vstack(XRActuals)              
        Epoch_Acc = accuracy_score(XRActuals,XRPredictions)
        print('Epoch: {}, XRLoss: {:.4f}, XRAccuracy: {:.4f}'.format(epoch, Epoch_Loss, Epoch_Acc))

        for i,(X,y) in enumerate(CTTrain):

            f  = X.shape[0]
            y  = y.reshape((len(y),1))
            y  = y.to(torch.float)
            T  = TaskembedCT(X)
            S  = SharedembedCT(X)
            F  = Generator(S)
            T  = TaskCT(T)
            C  = torch.cat((T,F),1)
            Output = ClassifierheadCT(C)
            Output = Output.to(torch.float)
            lossct   = Criterion_cct(Output, y)
            Optim_cct.zero_grad()
            CTLoss += lossct
            lossct.backward()
            Optim_cct.step()
            Output = torch.round(torch.sigmoid(Output))
            Output =  Output.cpu().detach().numpy()
            y =  y.cpu().detach().numpy()      
            CTActuals.append(y)
            CTPredictions.append(Output)

        Epoch_Loss = CTLoss/(len(CTTrain))
        CTPredictions, CTActuals = vstack(CTPredictions), vstack(CTActuals)              
        Epoch_Acc = accuracy_score(CTActuals,CTPredictions)
        print('Epoch: {}, CTLoss: {:.4f}, CTAccuracy: {:.4f}'.format(epoch, Epoch_Loss, Epoch_Acc))
        
        for i,(X,y) in enumerate(CLTrain):

            f  = X.shape[0]
            y  = y.reshape((len(y),1))
            y  = y.to(torch.float)
            T  = TaskembedCL(X)
            S  = SharedembedCL(X)
            F  = Generator(S)
            T  = TaskCL(T)
            C  = torch.cat((T,F),1)
            Output = ClassifierheadCL(C)
            Output = Output.to(torch.float)
            losscl   = Criterion_ccl(Output, y)
            Optim_ccl.zero_grad()
            CLLoss += losscl
            losscl.backward()
            Optim_ccl.step()
            Output = torch.round(torch.sigmoid(Output))
            Output =  Output.cpu().detach().numpy()
            y =  y.cpu().detach().numpy()      
            CLActuals.append(y)
            CLPredictions.append(Output)

        Epoch_Loss = CLLoss/(len(CLTrain))
        CLPredictions, CLActuals = vstack(CLPredictions), vstack(CLActuals)              
        Epoch_Acc = accuracy_score(CLActuals,CLPredictions)
        print('Epoch: {}, CLLoss: {:.4f}, CLAccuracy: {:.4f}'.format(epoch, Epoch_Loss, Epoch_Acc))
    print("Training Done")       
    #*********************************************************Evalution**********************************************************#   

    def Evaluate_XR():
        XRActuals = list()
        XRPredictions = list()
        for i,(X,y) in enumerate(XRTrain):
            
            f  = X.shape[0]
            y  = y.reshape((len(y),1))
            y  = y.to(torch.float)
            T  = TaskembedXR(X)
            T  = TaskXR(T)
            S  = SharedembedXR(X)
            F  = Generator(S)           
            C  = torch.cat((T,F),1)
            Output = ClassifierheadXR(C)
            Output = Output.to(torch.float)
            Output = torch.round(torch.sigmoid(Output))
            Output =  Output.cpu().detach().numpy()
            y      =  y.cpu().detach().numpy()      
            XRActuals.append(y)
            XRPredictions.append(Output)

        XRPredictions, XRActuals = vstack(XRPredictions), vstack(XRActuals)
        XRF1  = F1(XRActuals,XRPredictions)
        XRAcc = Acc(XRActuals,XRPredictions)
        XRSn  = recall_score(XRActuals,XRPredictions)
        XRSp  = Sp(XRActuals,XRPredictions)
        return XRF1,XRAcc,XRSn,XRSp

    def Evaluate_CT():       
        
        CTActuals = list()
        CTPredictions = list()
        for i,(X,y) in enumerate(CTTrain):
            f  = X.shape[0]
            y  = y.reshape((len(y),1))
            y  = y.to(torch.float)
            T  = TaskembedCT(X)
            T  = TaskCT(T)
            S  = SharedembedCT(X)
            F  = Generator(S)            
            C  = torch.cat((T,F),1)
            Output = ClassifierheadCT(C)
            Output = Output.to(torch.float)
            Output = torch.round(torch.sigmoid(Output))
            Output =  Output.cpu().detach().numpy()
            y      =  y.cpu().detach().numpy()      
            CTActuals.append(y)
            CTPredictions.append(Output)

        CTPredictions, CTActuals = vstack(CTPredictions), vstack(CTActuals)
        CTF1  = F1(CTActuals,CTPredictions)
        CTAcc = Acc(CTActuals,CTPredictions)
        CTSn  = recall_score(CTActuals,CTPredictions)
        CTSp  = Sp(CTActuals,CTPredictions)
        return CTF1,CTAcc,CTSn,CTSp
    def Evaluate_CL():       
        
        CLActuals = list()
        CLPredictions = list()
        for i,(X,y) in enumerate(CLTrain):
            f  = X.shape[0]
            y  = y.reshape((len(y),1))
            y  = y.to(torch.float)
            T  = TaskembedCL(X)
            T  = TaskCL(T)
            S  = SharedembedCL(X)
            F  = Generator(S)            
            C  = torch.cat((T,F),1)
            Output = ClassifierheadCL(C)
            Output = Output.to(torch.float)
            Output = torch.round(torch.sigmoid(Output))
            Output =  Output.cpu().detach().numpy()
            y      =  y.cpu().detach().numpy()      
            CLActuals.append(y)
            CLPredictions.append(Output)

        CLPredictions, CLActuals = vstack(CLPredictions), vstack(CLActuals)
        CLF1  = F1(CLActuals,CLPredictions)
        CLAcc = Acc(CLActuals,CLPredictions)
        CLSn  = recall_score(CLActuals,CLPredictions)
        CLSp  = Sp(CLActuals,CLPredictions)
        return CLF1,CLAcc,CLSn,CLSp
    
    XRF1,XRAcc,XRSn,XRSp = Evaluate_XR()
    print("XRays: F1score : {}, Accuracy: {},Specificity: {},Sensitivity: {}".format(XRF1,XRAcc,XRSp,XRSn))

    CTF1,CTAcc,CTSn,CTSp = Evaluate_CT()
    print("CT Scans: F1score : {}, Accuracy: {},Specificity: {},Sensitivity: {}".format(CTF1,CTAcc,CTSp,CTSn))
    
    CLF1,CLAcc,CLSn,CLSp = Evaluate_CL()
    print("Clinical Data : F1score : {}, Accuracy: {},Specificity: {},Sensitivity: {}".format(CLF1,CLAcc,CLSp,CLSn))
    
     #*********************************************************Thank You**********************************************************#   
    

if __name__ == "__main__":
    main()
