# -*- coding: utf-8 -*-
"""
Created on Wed May 27 12:02:17 2020

@author: Azeemushan
"""

#import all librarries here
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly
import plotly.express as px
import plotly.offline as pyo
from plotly.offline import init_notebook_mode,plot,iplot
import cufflinks as cf

pyo.init_notebook_mode(connected=True)
cf.go_offline()

#read csv file
df = pd.read_csv(r'C:\Users\Azeemushan\Desktop\Heart-Disease-Prediction-master\Heart-Disease-Prediction-master\heart.csv')

#info for shortforms to know what is what
info = ["age","1: male, 0: female","chest pain type, 1: typical angina, 2: atypical angina, 3: non-anginal pain, 4: asymptomatic","resting blood pressure"," serum cholestoral in mg/dl","fasting blood sugar > 120 mg/dl","resting electrocardiographic results (values 0,1,2)"," maximum heart rate achieved","exercise induced angina","oldpeak = ST depression induced by exercise relative to rest","the slope of the peak exercise ST segment","number of major vessels (0-3) colored by flourosopy","thal: 3 = normal; 6 = fixed defect; 7 = reversable defect"]
for i in range(len(info)):
    print(df.columns[i]+":\t\t\t"+info[i])

#see how many are one and how many are zero
df['target']
df.groupby('target').size() #we have 138 zero and 165 one

#for shape
df.shape

#size of dataframe
df.size

#describe for mean median
df.describe()

#check for null value
df.info()
#all values are nnon null so no preprocessing

#Now its time for visualization
df.hist(figsize= (14,14))
