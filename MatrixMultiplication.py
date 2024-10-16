# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 02:35:47 2024

@author: lewis
"""

import numpy as np

def gd(start, f, gradient, step_size, maxiter, tol=0.01):
    step = [start] ## tracking history of x
    x = start
    k=0
    for i in range(maxiter):
        diff = step_size*gradient(x)
        if np.all(np.abs(diff)<tol):
            
            break
        k=k+1
        x = x - diff
        fc = f(x)
        step.append(x) ## tracking
        
    
    return step, x

def f(x):
    b0, b1 = x
    return (1/6)*((3.8 - (b0 + 5*b1))**2 + (6.5 - (b0 - 6*b1))**2 + (11.5 - (b0 +7*b1))**2)

def grad_f(x):
    X = np.array([[1,5],
                  [1,6],
                  [1,7]])  # Add column of 1s for beta_0
    y = np.array([3.8,6.5,11.8])
    b0, b1 = x
    B = y - (X @ np.array([b0, b1]))
    beta_b0 = -(1/3) * np.sum(B)
    beta_b1 = -(1/3) * np.sum(B * np.array([5, 6, 7]))

    return np.array([beta_b0, beta_b1])



history, solution = gd(np.array([1,5]),f,grad_f,0.01,100)


#Problem 3 part 3 Matrix Multiplication with Backtracking

import numpy as np
import numpy.linalg as npl
from numpy import linalg as lp

def f(x):
    b0, b1 = x
    return (1/6)*((3.8 - (b0 + 5*b1))**2 + (6.5 - (b0 - 6*b1))**2 + (11.5 - (b0 +7*b1))**2)
    
def df(x):
    X = np.array([[1,5],
                  [1,6],
                  [1,7]])  # Add column of 1s for beta_0
    y = np.array([3.8,6.5,11.8])
    b0, b1 = x
    B = y - (X @ np.array([b0, b1]))
    beta_b0 = -(1/3) * np.sum(B)
    beta_b1 = -(1/3) * np.sum(B * np.array([5, 6, 7]))
    
    return np.array([beta_b0, beta_b1])
 
def step_size(x):
    alpha = 0.01
    beta = 0.5
    
    while f(x - alpha*df(x)) > (f(x) - 0.5*alpha*lp.norm(df(x))**2):
        alpha *= beta
        
    return alpha

def g(lambda_k,x,r):
    return f(x - lambda_k*r)

def steepestdescent(f,df,step_size,x0,tol=1.e-3,maxit=100):
    x = x0
    r = df(x0)
    iters = 0
    while ( np.abs(npl.norm(r))>tol and iters<maxit ):
        lambda_k = step_size(x)
        x = x - lambda_k * r
        r = df(x)
        iters += 1
    
    return x, iters

x0 = np.array([2.0,1.0])
x, iters =steepestdescent(f, df, step_size,x0, tol = 1.e-8, maxit = 7)