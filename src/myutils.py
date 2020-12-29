import numpy as np 
import cv2 as cv

def merge_box(r1,r2):
    x1,y1,w1,h1 = r1
    x2,y2,w2,h2 = r2
    x = x1 if x1 < x2 else x2
    y = y1 if y1 < y2 else y2
    u1 = x1 + w1
    v1 = y1 + h1
    u2 = x2 + w2
    v2 = y2 + h2
    u = u1 if u1>u2 else u2
    v = v1 if v1>v2 else v2
    w = u - x
    h = v - y
    if w <= (w1 + w2) and h <= (h1 + h2):
        return (x,y,w,h)
    return None

def merge_contours(cnt):
    new_cnt = []
    for r1 in cnt:
        tmp = None
        for i in range(len(new_cnt)):
            tmp = merge_box(r1,new_cnt[i])
            if tmp:
                new_cnt[i] = tmp
                break
        if not tmp:
            new_cnt.append(r1)
    return new_cnt

def filter_contours(w,h,width,height):
    if w*h < 50:
        return False
    return True

class_names = [str(i) for i in range(10)] + ["minus","plus","times","quote","t","y"]

STD_SIZE = (30,24)
VEC_SIZE = 30*24
