from sklearn.neural_network import MLPClassifier
import numpy as np
from myCNN import *
import pickle
import torch
import sys

data = np.loadtxt("../data/sym_data.dat")
labels = np.loadtxt("../data/sym_labels.dat")

if sys.argv[1] == "MLP":    
    model = MLPClassifier(hidden_layer_sizes=(200,50),max_iter=1000)
    model.fit(data,labels)
    
    fname = "../models/model_nn.pkl"
    with open(fname,"wb") as f:
        pickle.dump(model,f)

elif sys.argv[1] == "CNN":
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

