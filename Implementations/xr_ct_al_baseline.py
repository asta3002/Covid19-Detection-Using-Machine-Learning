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

    device =get_default_device()
    print(device)
    batch_size =10

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

    EfficientB0 = models.efficientnet_b0(pretrained=False).to(device)
    EfficientB1 = models.efficientnet_b1(pretrained=False).to(device)
    Resnet50    = models.resnet50(pretrained=True).to(device)
    EfficientB0 = nn.Sequential(*list(EfficientB0.children())[:-1])
    EfficientB1 = nn.Sequential(*list(EfficientB1.children())[:-1])
    Resnet50    = nn.Sequential(*list(Resnet50.children())[:-1])
    EfficientB0 = to_device(EfficientB0,device)
    EfficientB1 = to_device(EfficientB1,device)
    Resnet50    = to_device(Resnet50,device)
    Param_xr = []
    Param_ct = []
    def XTSK():
        T = nn.Sequential(
                EfficientB0(),
                nn.Linear(1280,960),
                nn.LeakyReLU(),
                nn.Linear(960,640),
                nn.LeakyReLU(),
                nn.Linear(640,960),
                nn.LeakyReLU(),
                nn.Linear(960,1280),
                nn.LeakyReLU()
                       )
        return T

    def TaskembedXR(X):
        f = X.shape[0]
        #X = EfficientB0(X)
        #X  = X.reshape((f,-1))
        TaskXR = XTSK()
        to_device(TaskXR,device)
        X = TaskXR(X)
        #Param_xr.append(EfficientB0.parameters())
        Param_xr.append(TaskXR.parameters())
        return X
    
    #Param_xr.append(TaskXR[0].weight)
    #Param_xr.append(TaskXR[2].weight)
    #Param_xr.append(TaskXR[4].weight)
    #Param_xr.append(TaskXR[6].weight)

    def CTSK():
        T = nn.Sequential(
                nn.Linear(1280,960),
                nn.LeakyReLU(),
                nn.Linear(960,640),
                nn.LeakyReLU(),
                nn.Linear(640,1),
                nn.LeakyReLU(),
                nn.Linear(960,1280),
                nn.LeakyReLU()
                       )
        return T

    def TaskembedCT(X):
        f = X.shape[0]
        X = EfficientB1(X)
        X  = X.reshape((f,-1))
        #TaskCT = CTSK()
        #to_device(TaskCT,device)
        #X = TaskCT(X)
        #Param_ct.append(TaskCT.parameters())
        return X
    #Param_ct.append(TaskCT[0].weight)
    #Param_ct.append(TaskCT[2].weight)
    #Param_ct.append(TaskCT[4].weight)
    #Param_ct.append(TaskCT[6].weight)

    def Sharedembed(X):
        
        f = X.shape[0]
        X = Resnet50(X)
        X  = X.reshape((f,-1))
        return X

    Generator = nn.Sequential(
        nn.Linear(2048,320),
        nn.LeakyReLU(),
        nn.Linear(320,160),
        nn.LeakyReLU(),
        nn.Linear(160,320),
        nn.LeakyReLU(),
        nn.Linear(320,2048)
    )
    Param_G =[]
    Param_G.append(Generator[0].weight)
    Param_G.append(Generator[2].weight)
    Param_G.append(Generator[4].weight)
    Param_G.append(Generator[6].weight)
    
    Generator = to_device(Generator,device)
    optim_G = torch.optim.Adam(Param_G, lr=0.001)
    optim_Gf = torch.optim.Adam(Param_G, lr=0.00001)
    
    def G_Loss(GenEmbed,InvLabel):
        T_L = torch.nn.functional.binary_cross_entropy_with_logits(GenEmbed, InvLabel)
        
        return T_L

    Discriminator =nn.Sequential(
        nn.Linear(2048,320),
        nn.LeakyReLU(),
        nn.Linear(320,160),
        nn.LeakyReLU(),
        nn.Linear(160,1),

    )
    Param_D =[]
    Param_D.append(Discriminator[0].weight)
    Param_D.append(Discriminator[2].weight)
    Param_D.append(Discriminator[4].weight)
    
    optim_D = torch.optim.Adam(Param_D, lr=0.001)
    optim_Df = torch.optim.Adam(Param_D, lr=0.00001)
    Discriminator = to_device(Discriminator,device)

    

    def D_Loss(ActEmbed,GenEmbed,Label,InvLabel):
               
        A_L = torch.nn.functional.binary_cross_entropy_with_logits(ActEmbed, Label)
        F_L = torch.nn.functional.binary_cross_entropy_with_logits(GenEmbed, InvLabel)
        T_L = A_L +F_L
        return T_L

    

    ClassifierheadXR = nn.Sequential(
            nn.Dropout(p=0.2),                   
            nn.Linear(3328,320),
            nn.LeakyReLU(),            
            nn.Linear(320,160),
            nn.LeakyReLU(),
            nn.Linear(160,1)


        )
    Param_xr.append(ClassifierheadXR[1].weight)
    Param_xr.append(ClassifierheadXR[3].weight)
    Param_xr.append(ClassifierheadXR[5].weight)
    

    ClassifierheadCT = nn.Sequential(
            nn.Dropout(p=0.2),                    
            nn.Linear(3328,320),
            nn.LeakyReLU(),
            nn.Linear(320,160),
            nn.LeakyReLU(),
            nn.Linear(160,1)


        )
    Param_ct.append(ClassifierheadCT[1].weight)
    Param_ct.append(ClassifierheadCT[3].weight)
    Param_ct.append(ClassifierheadCT[5].weight)
    
    

    ClassifierheadXR = to_device(ClassifierheadXR,device)
    ClassifierheadCT = to_device(ClassifierheadCT,device)

    criterion_xr = nn.BCEWithLogitsLoss()
    criterion_ct = nn.BCEWithLogitsLoss()
    optim_XR = torch.optim.Adam(Param_xr, lr=0.001)
    optim_CT = torch.optim.Adam(Param_ct, lr=0.001)
    optim_XRf = torch.optim.Adam(Param_xr, lr=0.00001)
    optim_CTf = torch.optim.Adam(Param_ct, lr=0.00001)
    epochs = 50
    def train_D():
                       
        for epoch in range(10):
            DLoss =0
            for i,(X,y) in enumerate(XRTrain):
                                
                f = X.shape[0]
                y  = y.reshape((len(y),1))
                y  = y.to(torch.float)
                
                optim_D.zero_grad()                
                
                S = Sharedembed(X)
                
                F = Generator(S)
                
                R_O = Discriminator(S)
                F_O = Discriminator(F)
                Label  = torch.ones(f,1).to(device)
                InvLabel = torch.zeros(f,1).to(device)
                            
                               
                
                dloss = D_Loss(R_O,F_O,Label,InvLabel)          
                dloss.backward()
                optim_D.step()
                DLoss += dloss
            DLoss  =  (DLoss)/(len(XRTrain))
                               
            print(DLoss)
                
            for i,(X,y) in enumerate(CTTrain):
                f = X.shape[0]
                y  = y.reshape((len(y),1))
                y  = y.to(torch.float)               
                
                optim_D.zero_grad()
                S = Sharedembed(X)
                F = Generator(S)
                
                R_O = Discriminator(S)
                F_O = Discriminator(F)
                InvLabel  = torch.ones(f,1).to(device)
                Label = torch.zeros((f,1)).to(device)
                
                
                
                dloss = D_Loss(R_O,F_O,Label,InvLabel)                
                dloss.backward()                
                optim_D.step()
                
    def fine_tune_D():
                               
        for epoch in range(5):
            DLoss =0
            
            
            for i,(X,y) in enumerate(XRTrain):
                                
                f = X.shape[0]
                y  = y.reshape((len(y),1))
                y  = y.to(torch.float)
                
                optim_Df.zero_grad()                
                
                S = Sharedembed(X)
                F = Generator(S)
                R_O = Discriminator(S)
                F_O = Discriminator(F)
                Label  = torch.ones(f,1).to(device)
                InvLabel = torch.zeros(f,1).to(device)
                            
                                 
                
                dloss = D_Loss(R_O,F_O,Label,InvLabel)          
                dloss.backward()
                optim_Df.step()
                DLoss += dloss
            DLoss  = (DLoss)/(len(XRTrain))
            print(DLoss)
                
            for i,(X,y) in enumerate(CTTrain):
                
                f = X.shape[0]
                y  = y.reshape((len(y),1))
                y  = y.to(torch.float)
                
                optim_Df.zero_grad()
                S = Sharedembed(X)
                F = Generator(S)
                R_O = Discriminator(S)
                F_O = Discriminator(F)
                InvLabel  = torch.ones(f,1).to(device)
                Label = torch.zeros(f,1).to(device)
                     
                
                
                dloss = D_Loss(R_O,F_O,Label,InvLabel)
                
                dloss.backward()
                
                optim_Df.step()     
    def train_G():
                       
        for epoch in range(10):
            
            for i,(X,y) in enumerate(XRTrain):
                                
                f = X.shape[0]
                y  = y.reshape((len(y),1))
                y  = y.to(torch.float)
                
                optim_G.zero_grad()                
                
                S = Sharedembed(X)
                F = Generator(S)
                R_O = Discriminator(S)
                F_O = Discriminator(F)
                Label  = torch.ones(f,1).to(device)
                InvLabel = torch.zeros(f,1).to(device)
                            
                                 
                
                gloss = G_Loss(F_O,InvLabel)         
                gloss.backward()
                optim_G.step()
                
            for i,(X,y) in enumerate(CTTrain):
                
                f = X.shape[0]
                y  = y.reshape((len(y),1))
                y  = y.to(torch.float)
                
                optim_G.zero_grad()
                S = Sharedembed(X)
                F = Generator(S)
                R_O = Discriminator(S)
                F_O = Discriminator(F)
                InvLabel  = torch.ones(f,1).to(device)
                Label = torch.zeros(f,1).to(device)
                     
                
                
                gloss = G_Loss(F_O,InvLabel)         
                gloss.backward()
                optim_G.step()
                
    def fine_tune_G():
                               
        for epoch in range(5):
            
            for i,(X,y) in enumerate(XRTrain):
                                
                f = X.shape[0]
                y  = y.reshape((len(y),1))
                y  = y.to(torch.float)
                
                optim_Gf.zero_grad()                
                
                S = Sharedembed(X)
                F = Generator(S)
                R_O = Discriminator(S)
                F_O = Discriminator(F)
                Label  = torch.ones(f,1).to(device)
                InvLabel = torch.zeros(f,1).to(device)
                            
                                 
                
                gloss = G_Loss(F_O,InvLabel)         
                gloss.backward()
                optim_Gf.step()
                
            for i,(X,y) in enumerate(CTTrain):
                
                f = X.shape[0]
                y  = y.reshape((len(y),1))
                y  = y.to(torch.float)
                optim_Gf.zero_grad()
                S = Sharedembed(X)
                
                F = Generator(S)
                R_O = Discriminator(S)
                F_O = Discriminator(F)
                InvLabel  = torch.ones(f,1).to(device)
                Label = torch.zeros(f,1).to(device)
                     
                
                
                gloss = G_Loss(F_O,InvLabel)         
                gloss.backward()
                optim_Gf.step()   
    def train_task_specific_exctractors(epochs):
        for epoch in range(epochs):
            
            for i,(X,y) in enumerate(XRTrain):
                f = X.shape[0]
                y  = y.reshape((len(y),1))
                y  = y.to(torch.float)
                T = TaskembedXR(X)
                
            
            
    def train_step(epochs):
                
        for epoch in range(epochs):
            XRLoss =0
            XRActuals = list()
            XRPredictions = list()
            CTLoss =0
            CTActuals = list()
            CTPredictions = list()
            for i,(X,y) in enumerate(XRTrain):
                
                f = X.shape[0]
                y  = y.reshape((len(y),1))
                y  = y.to(torch.float)
                
                T = TaskembedXR(X)
                
                S = Sharedembed(X)
                F = Generator(S)
                
                
                
                
                C = torch.cat((T,F),1)
                Output = ClassifierheadXR(C)
                Output = Output.to(torch.float)
                optim_XR.zero_grad()
                loss = criterion_xr(Output, y)
                loss.backward()
                 
                
                XRLoss += loss
                
                optim_XR.step()
               
                Output = torch.round(torch.sigmoid(Output))
                Output =  Output.cpu().detach().numpy()
                y =  y.cpu().detach().numpy()      
                XRActuals.append(y)
                XRPredictions.append(Output)
                
            Epoch_Loss = XRLoss/(len(XRTrain))
            XRPredictions, XRActuals = vstack(XRPredictions), vstack(XRActuals)              
            Epoch_Acc = accuracy_score(XRActuals,XRPredictions)
            print('Epoch: {}, XRLoss: {:.4f}, XRAccuracy: {:.4f}'.format(epoch, Epoch_Loss, Epoch_Acc))

            for i,(X,y) in enumerate(CTTrain):
                
                f = X.shape[0]
                y  = y.reshape((len(y),1))
                y  = y.to(torch.float)
                
                T  = TaskembedCT(X)
                
                S = Sharedembed(X)
                F = Generator(S)
                
                InvLabel  = torch.ones(f,1).to(device)
                Label = torch.zeros(f,1).to(device)
                
                
                    
                C = torch.cat((T,F),1)
                Output = ClassifierheadCT(C)
                Output = Output.to(torch.float)
                loss = criterion_ct(Output, y)
                optim_CT.zero_grad()
                
                
                
                CTLoss += loss
                
                loss.backward()
                optim_CT.step()
                
                Output = torch.round(torch.sigmoid(Output))
                Output =  Output.cpu().detach().numpy()
                y =  y.cpu().detach().numpy()      
                CTActuals.append(y)
                CTPredictions.append(Output)
                
            Epoch_Loss = CTLoss/(len(CTTrain))
            CTPredictions, CTActuals = vstack(CTPredictions), vstack(CTActuals)              
            Epoch_Acc = accuracy_score(CTActuals,CTPredictions)
            print('Epoch: {}, CTLoss: {:.4f}, CTAccuracy: {:.4f}'.format(epoch, Epoch_Loss, Epoch_Acc))
            
    def fine_tune_step():
                
        for epoch in range(30):
            XRLoss =0
            XRActuals = list()
            XRPredictions = list()
            CTLoss =0
            CTActuals = list()
            CTPredictions = list()
            for i,(X,y) in enumerate(XRTrain):
                
                f = X.shape[0]
                y  = y.reshape((len(y),1))
                y  = y.to(torch.float)
                
                T = TaskembedXR(X)
                
                S = Sharedembed(X)
                F = Generator(S)
                
                
                
                
                C = torch.cat((T,F),1)
                Output = ClassifierheadXR(C)
                Output = Output.to(torch.float)
                optim_XRf.zero_grad()
                loss = criterion_xr(Output, y)
                loss.backward()
                 
                
                XRLoss += loss
                
                optim_XRf.step()
               
                Output = torch.round(torch.sigmoid(Output))
                Output =  Output.cpu().detach().numpy()
                y =  y.cpu().detach().numpy()      
                XRActuals.append(y)
                XRPredictions.append(Output)
                
            Epoch_Loss = XRLoss/(len(XRTrain))
            XRPredictions, XRActuals = vstack(XRPredictions), vstack(XRActuals)              
            Epoch_Acc = accuracy_score(XRActuals,XRPredictions)
            print('Epoch: {}, XRLoss: {:.4f}, XRAccuracy: {:.4f}'.format(epoch, Epoch_Loss, Epoch_Acc))

            for i,(X,y) in enumerate(CTTrain):
                
                f = X.shape[0]
                y  = y.reshape((len(y),1))
                y  = y.to(torch.float)
                
                T  = TaskembedCT(X)
                
                S = Sharedembed(X)
                F = Generator(S)
                
                InvLabel  = torch.ones(f,1).to(device)
                Label = torch.zeros(f,1).to(device)
                
                
                    
                C = torch.cat((T,F),1)
                Output = ClassifierheadCT(C)
                Output = Output.to(torch.float)
                loss = criterion_ct(Output, y)
                optim_CTf.zero_grad()
                
                
                
                CTLoss += loss
                
                loss.backward()
                optim_CTf.step()
                
                Output = torch.round(torch.sigmoid(Output))
                Output =  Output.cpu().detach().numpy()
                y =  y.cpu().detach().numpy()      
                CTActuals.append(y)
                CTPredictions.append(Output)
                
            Epoch_Loss = CTLoss/(len(CTTrain))
            CTPredictions, CTActuals = vstack(CTPredictions), vstack(CTActuals)              
            Epoch_Acc = accuracy_score(CTActuals,CTPredictions)
            print('Epoch: {}, CTLoss: {:.4f}, CTAccuracy: {:.4f}'.format(epoch, Epoch_Loss, Epoch_Acc))

    def Evaluate_XR():
        XRActuals = list()
        XRPredictions = list()
        for i,(X,y) in enumerate(XRTrain):
            
            f = X.shape[0]
            y  = y.reshape((len(y),1))
            y  = y.to(torch.float)
            T = TaskembedXR(X)
            
            S = Sharedembed(X)
            F = Generator(S)
           
            C = torch.cat((T,F),1)
            Output = ClassifierheadXR(C)
            Output = Output.to(torch.float)
            Output = torch.round(torch.sigmoid(Output))
            Output =  Output.cpu().detach().numpy()
            y =  y.cpu().detach().numpy()      
            XRActuals.append(y)
            XRPredictions.append(Output)

        XRPredictions, XRActuals = vstack(XRPredictions), vstack(XRActuals)
        XRF1 = F1(XRActuals,XRPredictions)
        XRAcc = Acc(XRActuals,XRPredictions)
        XRSn = recall_score(XRActuals,XRPredictions)
        XRSp = Sp(XRActuals,XRPredictions)
        return XRF1,XRAcc,XRSn,XRSp

    def Evaluate_CT():       
        
        CTActuals = list()
        CTPredictions = list()
        for i,(X,y) in enumerate(CTTrain):
            f = X.shape[0]
            y  = y.reshape((len(y),1))
            y  = y.to(torch.float)
            T = TaskembedCT(X)
            
            S = Sharedembed(X)
            F = Generator(S)
            
            C = torch.cat((T,F),1)
            Output = ClassifierheadCT(C)
            Output = Output.to(torch.float)
            Output = torch.round(torch.sigmoid(Output))
            Output =  Output.cpu().detach().numpy()
            y =  y.cpu().detach().numpy()      
            CTActuals.append(y)
            CTPredictions.append(Output)

        CTPredictions, CTActuals = vstack(CTPredictions), vstack(CTActuals)
        CTF1 = F1(CTActuals,CTPredictions)
        CTAcc = Acc(CTActuals,CTPredictions)
        CTSn = recall_score(CTActuals,CTPredictions)
        CTSp = Sp(CTActuals,CTPredictions)
        return CTF1,CTAcc,CTSn,CTSp
    """for i, (X,y) in enumerate(XRTrain):
        X = TaskembedXR(X)
        
        print(X.shape)
        break;"""
    #train_D()
    #fine_tune_D()
    #train_G()
    #fine_tune_G()
    train_step(epochs)
    fine_tune_step()
    XRF1,XRAcc,XRSn,XRSp = Evaluate_XR()
    print("XRays: F1score : {}, Accuracy: {},Specificity: {},Sensitivity: {}".format(XRF1,XRAcc,XRSp,XRSn))

    CTF1,CTAcc,CTSn,CTSp = Evaluate_CT()
    print("CT Scans: F1score : {}, Accuracy: {},Specificity: {},Sensitivity: {}".format(CTF1,CTAcc,CTSp,CTSn))

if __name__ == "__main__":
    main()