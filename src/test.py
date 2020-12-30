import numpy as np
import cv2 as cv
import pickle
from myutils import *
from myCNN import *
import torch
import sys

if sys.argv[1] == "MLP":
    model_name = "../models/model_nn.pkl"
    with open(model_name,"rb") as f:
        model = pickle.load(f)
elif sys.argv[1] == "CNN":
    model_name = "../models/model_cnn_torch"
    model = CNN()
    model.load_state_dict(torch.load(model_name))

img = cv.imread("../data/test2.jpg",0)
ret, thresh = cv.threshold(img,127,255,0)
contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
cnt = [cv.boundingRect(e) for e in contours[1:]]
cnt = merge_contours(cnt)
eq = ""
elems = []
for e in cnt:
    x,y,w,h = e
    img = cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),1)
    crop = cv.resize(thresh[y:y+h,x:x+w],STD_SIZE)/255.0
    if sys.argv[1] == "MLP":
        crop = crop.reshape((1,VEC_SIZE))
    elif sys.argv[1] == "CNN":
        crop = crop.reshape([1,1,STD_SIZE[0],STD_SIZE[1]])
        crop = torch.from_numpy(crop).float()
    pred = model.predict(crop)
    img = cv.putText(img,class_names[int(pred[0])],(x,y),cv.FONT_ITALIC,0.5,(0,255,0))
    elems.append([x,y,w,h,pred[0]])
for s in sorted(elems,key=lambda x : x[0]):
    eq += class_names[int(s[-1])]

eq = eq.replace("quote","'")
eq = eq.replace("minus","-")
eq = eq.replace("plus","+")
eq = eq.replace("times","x")
eq = eq.replace("--","=")

print(eq)
cv.imshow("View",img)
cv.waitKey(0)

