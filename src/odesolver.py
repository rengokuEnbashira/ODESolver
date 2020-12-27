import numpy as np
import matplotlib.pyplot as plt
from math import *

_t = np.linspace(0,1,100)
_maxIt = 100
_prec = 1e-5
_h = 0.001

def rungeKutta(fun_f,alpha,t=_t):
    y = [alpha]
    N = len(t)
    for i in range(1,N):
        h = t[i] - t[i-1]
        k1 = fun_f(y[i-1],t[i-1])
        k2 = fun_f(y[i-1] + h*k1/2,t[i-1] + h/2)
        k3 = fun_f(y[i-1] + h*k2/2,t[i-1] + h/2)
        k4 = fun_f(y[i-1] + h*k3, t[i-1] + h)
        dy = h*(k1 + 2*k2 + 2*k3 + k4)/6
        y.append(y[i-1] + dy)
    return np.array(y)

def newton_method(fun_f,fun_df,x0,maxIt=_maxIt,prec=_prec):
    x = x0
    for i in range(maxIt):
        f  = fun_f(x)
        df = fun_df(x)
        x1 = x - f/df
        if np.abs(x1 - x) < prec*np.abs(x):
            break
        x = x1
    return x

"""
y' + y'' + t*y = sin(t)
"""

class MyEquation:
    def __init__(self,str_eq):
        tmp = str_eq.split("=")
        self.right = tmp[0]
        self.left = tmp[1]
        self.equ = self.right + " - (" + self.left + " )"
    def fun_f(self,y,t):
        tmp_f = self.equ
        
        if 't' in tmp_f:
            tmp_f = tmp_f.replace('t',str(t))

        _y = "y" + "'"*len(y)
        tmp_f = tmp_f.replace(_y,"u")
        out = []
        for i in y[-1:0:-1]:
            _y = _y[:-1]
            if _y in tmp_f:
                tmp_f = tmp_f.replace(_y,str(i))
            out.insert(0,i)
        if 'y' in tmp_f:
            tmp_f = tmp_f.replace('y',str(y[0]))
        f = lambda x : eval(tmp_f.replace("u",str(x)))
        df = lambda x : (f(x + _h) - f(x))/_h
        x0 = 0.01
        
        dy =  newton_method(f,df,x0)
        out.append(dy)

        return np.array(out)
    def solve(self,alpha,t=_t):
        return rungeKutta(self.fun_f,alpha,t=t)

"""
f = lambda x : x**2 - 5
df = lambda x : 2*x

print(newton_method(f,df,0.1),5**0.5)
"""

"""
f = lambda y,t : np.array([y[1],-y[0]])
alpha = np.array([0,1])
t = np.linspace(0,20,100)
y = rungeKutta(f,alpha,t=t)
plt.plot(y[:,0])
plt.show()
print(y)
"""
eq = MyEquation("y''' + y + t*y'= t*sin(2*t)")

alpha = np.array([0,0,1])
t = np.linspace(0,10,100)

y = eq.solve(alpha,t=t)
plt.plot(y[:,0])
plt.show()
print(y)
