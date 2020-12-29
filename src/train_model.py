from sklearn.neural_network import MLPClassifier
import numpy as np
import pickle

data = np.loadtxt("../data/sym_data.dat")
labels = np.loadtxt("../data/sym_labels.dat")

model = MLPClassifier(hidden_layer_sizes=(200,50),max_iter=1000)
model.fit(data,labels)

fname = "../models/model_nn.pkl"
with open(fname,"wb") as f:
    pickle.dump(model,f)
