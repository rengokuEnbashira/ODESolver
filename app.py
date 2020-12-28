from flask import Flask, render_template,request
from src.odesolver import *
import numpy as np

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/solve",methods=["POST"])
def solve():
    if request.method == "POST":
        response = {"data":"test"}
        eq = MyEquation("y'' + 5*y' - 4*y = sin(10*t)")

        alpha = np.array([0,0])
        t = np.linspace(0,3,100)
        
        y = eq.solve(alpha,t=t)
        print(y)
        
        print(request.form["data"])
        return response

if __name__ == "__main__":
    app.run("localhost",5050)
