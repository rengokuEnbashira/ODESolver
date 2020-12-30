import torch
import torch.nn as nn
import numpy as np 

class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2,stride=2)

        self.conv_layer1 = nn.Conv2d(1,20,3)
        self.conv_layer2 = nn.Conv2d(20,40,3)

        self.fc_layer1 = nn.Linear(40*6*4,80)
        self.fc_layer2 = nn.Linear(80,16)

        self.softmax = nn.Softmax(dim=1)
        
    def forward(self,x):
        x = self.conv_layer1(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv_layer2(x)
        x = self.relu(x)
        x = self.pool(x)

        x = x.view(x.size()[0],-1)
        
        x = self.fc_layer1(x)
        x = self.relu(x)

        x = self.fc_layer2(x)
        return x

    def predict_prob(self,x):
        x = self.forward(x)
        return self.softmax(x)

    def predict(self,x):
        p = self.predict_prob(x)
        return torch.argmax(p,axis=1)
    
    def train(self,train_loader,optimizer,criterion, epochs):
        for epoch in range(epochs):
            for x,y in train_loader:
                optimizer.zero_grad()
                tmp_o = self.forward(x)
                loss = criterion(tmp_o,y)
                loss.backward()
                optimizer.step()

if __name__ == "__main__":
    cnn = CNN()

    BATCH_SIZE = 20
    LEARNING_RATE = 0.01
    MOMENTUM = 0.9
    WEIGHT_DECAY = 0.005
    
    data = np.loadtxt("../data/sym_data.dat")
    labels = np.loadtxt("../data/sym_labels.dat")
    
    N = len(data)

    data = data.reshape([N,1,30,24])
    data = torch.from_numpy(data).float()

    labels = torch.from_numpy(labels).long()

    train_set = torch.utils.data.TensorDataset(data,labels)
    train_loader = torch.utils.data.DataLoader(train_set,batch_size=BATCH_SIZE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(cnn.parameters(),lr=LEARNING_RATE,momentum=MOMENTUM,weight_decay=WEIGHT_DECAY)

    cnn.train(train_loader,optimizer,criterion,200)

    torch.save(cnn.state_dict(),"../models/model_cnn_torch")
    
    L = cnn.predict(data[10:30])
    print(L)
    print(labels[10:30])
