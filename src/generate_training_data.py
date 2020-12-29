import numpy as np
import cv2 as cv
from myutils import *

w1 = 0
h1 = 0
data = []
labels = []
for i,f in enumerate(class_names):
    img = cv.imread("../data/%s.jpg"%(f),0)
    height, width = img.shape
    ret,thresh = cv.threshold(img,127,255,0)
    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    cnt = [cv.boundingRect(e) for e in contours[1:]]
    cnt = merge_contours(cnt)
    for e in cnt:
        x,y,w,h = e
        if filter_contours(w,h,width,height):
            img1 = cv.resize(thresh[y:y+h,x:x+w],STD_SIZE)
            data.append(img1.reshape(VEC_SIZE)/255.0)
            labels.append(i)

data = np.array(data)
labels = np.array(labels)

np.savetxt("../data/sym_data.dat",data)
np.savetxt("../data/sym_labels.dat",labels)

print(data.shape)
print(labels.shape)

