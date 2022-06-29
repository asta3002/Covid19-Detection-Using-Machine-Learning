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

    transform = transforms.Compose([transforms.ToTensor()]) # Rescales all images to betwee 0 and 1

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
    batch_size =4

    EfficientB0  = models.efficientnet_b0(pretrained=True).to(device)
    EfficientB0

    def set_parameter_requires_grad(model, feature_extracting):
        
        if feature_extracting:
            for param in model.parameters():
                param.requires_grad = False

    def truth_parameter_requires_grad(model):
      for param in model.parameters():
         param.requires_grad = True

    set_parameter_requires_grad(EfficientB0,feature_extracting=True)

    EfficientB0.classifier = nn.Sequential(
        
     Linear(in_features=1280, out_features=512, bias=True),
    nn.SiLU(),
    nn.Dropout(p=0.5),
    nn.Linear(512,256),
    nn.SiLU(),
    nn.Dropout(p=0.5),
    nn.Linear(256,1),
    )

    cnt1 =0
    cnt2 =0
    for (p) in EfficientB0.parameters():
        cnt1 =cnt1+1
        if p.requires_grad == True:
            
            cnt2 = cnt2+1
    print(cnt1,cnt2)

    RPATH = r'/content/drive/MyDrive/IIT Patna Internship 2021/Embeddings Dataset'

    XRdatasettrain =  ImageFolder(r"./X-Ray Train", transform=transform)
    XRdatasettest  = ImageFolder(r"./X-Ray Test",transform =transform)
    XRTrain = DataLoader(XRdatasettrain, batch_size,shuffle =True)
    XRTest = DataLoader(XRdatasettest, batch_size,shuffle =True)
    XRTrain = DeviceDataLoader(XRTrain, device)
    XRTest = DeviceDataLoader(XRTest, device)

    Criterion = nn.BCEWithLogitsLoss()
    epochs1  =50
    lr = 0.001

    feature_extract = True

    """params_to_update = EfficientB0.parameters()
    cnt3=0
    if feature_extract:
        params_to_update = []
        for name,param in EfficientB0.named_parameters():
            
            if param.requires_grad == True:
              cnt3 =cnt3 +1         
              params_to_update.append(param)
                

    print(cnt3)
    # Observe that all parameters are being optimized"""



    EfficientB0 = to_device(EfficientB0,device)

    def train_model(model, dataloaders, criterion, optimizer, num_epochs=epochs1):
        for epoch in range(num_epochs):
            
            Loss = 0
            Acc =0
            XRActuals = list()
            XRPredictions = list()
            for i,(X,y) in enumerate(XRTrain):
                y  = y.reshape((len(y),1))
                y  = y.to(torch.float)
                  #print(i)
                optimizer.zero_grad()
                outputs = model(X)
                  
                  
                outputs = outputs.to(torch.float)
                loss = criterion(outputs, y)
                Loss += loss
                outputs = torch.round(torch.sigmoid(outputs))
                outputs =  outputs.cpu().detach().numpy()
                y =  y.cpu().detach().numpy()
                  
                  
                XRActuals.append(y)
                XRPredictions.append(outputs)
                loss.backward()
                optimizer.step()
                  #break;
                
            Epoch_Loss = Loss/(len(XRTrain))
            XRPredictions, XRActuals = vstack(XRPredictions), vstack(XRActuals)              
            Epoch_Acc = accuracy_score(XRActuals,XRPredictions)
            print('Epoch: {}, Loss: {:.4f}, Accuracy: {:.4f}'.format(epoch, Epoch_Loss, Epoch_Acc))

    optimizer_ft =  torch.optim.Adam(EfficientB0.parameters(), lr=lr)

    train_model(EfficientB0,XRTrain,Criterion,optimizer_ft)

    epochsf =30
    lrf = 1e-6
    Criterion1 = nn.BCEWithLogitsLoss()

    #finetuning
    truth_parameter_requires_grad(EfficientB0)
    cnt1 =0
    cnt2 =0
    for (p) in EfficientB0.parameters():

        cnt1 =cnt1+1
        if p.requires_grad == True:
            
            cnt2 = cnt2+1
        print(cnt1,cnt2)

    EfficientB0 = to_device(EfficientB0,device)
    optimizer = torch.optim.Adam(EfficientB0.parameters(), lr=lrf,betas=(0.9,0.999))



    train_model(EfficientB0,XRTrain,Criterion1,optimizer,epochsf)

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

    def evaluate_XR(model,test_dl):
        XRActuals = list()
        XRPredictions = list()
        for i,(X,y) in enumerate(test_dl):
            y = y.reshape((len(y)),1)
            y = y.to(torch.float)
            Y = y.cpu().numpy()
            Y = Y.reshape(len(Y),1)
            
            XRActuals.append(Y)
            X  =  model(X)
            X = X.to(torch.float)
            X = torch.round(torch.sigmoid(X))
            
            
            #print(P)
            
           # P = P.reshape((len(P),1))
            X = X.cpu().detach().numpy()
            
            XRPredictions.append(X)
      
        XRPredictions, XRActuals = vstack(XRPredictions), vstack(XRActuals)
          
        XRF1 = F1(XRActuals,XRPredictions)
        XRAcc = Acc(XRActuals,XRPredictions)
        XRSn = recall_score(XRActuals,XRPredictions)
        XRSp = Sp(XRActuals,XRPredictions)
        return XRF1,XRAcc,XRSn,XRSp

    XRF1,XRAcc,XRSn,XRSp = evaluate_XR(EfficientB0,XRTest)
    print("XRays: F1score : {}, Accuracy: {},Specificity: {},Sensitivity: {}".format(XRF1,XRAcc,XRSp,XRSn))
if __name__ == "__main__":
    main()
