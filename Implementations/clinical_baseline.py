
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

    batch_size = 32
    def prepare_data(path):
        
        # load the dataset
        dataset = CSVDataset(path)
        dataset = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        #dataset = torch.tensor(dataset, dtype = torch.float32)
        # prepare data loaders
        
        return dataset

    CLdatasettrain  = prepare_data(r"./TrainClinical.csv")
    CLdatasettest = prepare_data(r"./TestClinical.csv")
    CLTrain = DeviceDataLoader(CLdatasettrain, device)
    CLTest = DeviceDataLoader(CLdatasettest, device)

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

    model = MLP(8)
    model = to_device(model,device)
    T =[]
    
    def G():
        S  =  nn.Sequential(
                        nn.Linear(1280,960),
                        nn.LeakyReLU(),
                        nn.Linear(960,640),
                        nn.LeakyReLU(),
                        #nn.BatchNormalization(),           
                        nn.Linear(640,1),
                               )
        T.append(S[0].weight)
        T.append(S[2].weight)
        T.append(S[4].weight)
        return S
    P = G()
    print(P.parameters())
    
    optim_CT = torch.optim.Adam(T, lr=0.001)

    F = list(model.children())[:-2]
    print(F)

    epochs = 50
    lr = 0.001



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

    def train_model(train_dl, model,epochs):
        
        # define the optimization
        criterion = nn.BCEWithLogitsLoss()
        optimizer = Adam(model.parameters(), lr=lr)
        # enumerate epochs
        for epoch in range(epochs):
            Loss = 0
            CLActuals = list()
            CLPredictions = list()
            for i,(X,y) in enumerate(train_dl):
                
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
                CLActuals = np.append(CLActuals,y)
                CLPredictions = np.append(CLPredictions,outputs)
                loss.backward()
                optimizer.step()  #break;
                #print(i)
              
            Epoch_Loss = Loss/(len(train_dl))
            CLPredictions, CLActuals = vstack(CLPredictions), vstack(CLActuals)              
            Epoch_Acc = accuracy_score(CLActuals,CLPredictions)
            print('Epoch: {}, Loss: {:.4f}, Accuracy: {:.4f}'.format(epoch, Epoch_Loss, Epoch_Acc))

    # evaluate the model
    def evaluate_model(test_dl, model):
        
        CLPredictions, CLActuals = list(), list()
        for i, (inputs, targets) in enumerate(test_dl):
            yhat = model(inputs)
            yhat = torch.round(torch.sigmoid(yhat))
             # retrieve numpy array
            yhat = yhat.cpu().detach().numpy()
              
            actual = targets.cpu().detach().numpy()
            actual = actual.reshape((len(actual), 1))
                # round to class values
            yhat = yhat.round()
                
                
                # store
            CLActuals = np.append(CLActuals,actual)
            CLPredictions = np.append(CLPredictions,yhat)
        CLpredictions, CLActuals = vstack(CLPredictions), vstack(CLActuals)
        CLF1 = F1(CLActuals,CLPredictions)
        CLAcc = Acc(CLActuals,CLPredictions)
        CLSn = recall_score(CLActuals,CLPredictions)
        CLSp = Sp(CLActuals,CLPredictions)
        return CLF1,CLAcc,CLSn,CLSp

    train_model(CLTrain, model,epochs)

    CLF1,CLAcc,CLSn,CLSp = evaluate_model(CLTest,model)
    print("CL data: F1score : {}, Accuracy: {},Specificity: {},Sensitivity: {}".format(CLF1,CLAcc,CLSp,CLSn))

    PATH = r'/content/drive/MyDrive/CLModel.pt'
    torch.save(model.state_dict(), PATH)

if __name__ == "__main__":
    main()






