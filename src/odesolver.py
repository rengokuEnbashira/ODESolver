import numpy as np
import matplotlib.pyplot as plt

class MyEquation:
    def __init__(self,str_eq):
        tmp = str_eq.split("=")
        self.right = tmp[0]
        self.left = tmp[1]
        self.fun_f = self.left + " - (" + self.right + ")"
    def eval(self,inp):
        pass
def rungeKutta(eq,alpha=)
