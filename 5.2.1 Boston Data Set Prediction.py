import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 

from sklearn.datasets import load_boston
dataset  =  load_boston()
X = dataset.data
y = dataset.target


from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,y)

lin_reg.score(X,y)

lin_reg.coef_
lin_reg.intercept_

# dont know what to do with this lin_reg.predict(X[6:9,:])

y_pred = lin_reg.predict(X)