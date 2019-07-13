import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#titanic dataset pe  bhi lagana hai
from sklearn.datasets import load_breast_cancer
datasets = load_breast_cancer()

X = datasets.data
y = datasets.target

#from sklearn.model_selection import train_test_split
#X_train, X_test , y_train , y_test = train_test_split(X,y)

from sklearn.svm import SVC
svm = SVC() 
svm.fit(X,y)

svm.score(X,y)



from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression()



from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()

from sklearn.tree import DecisionTreeClassifier
dtf = DecisionTreeClassifier()

from sklearn.svm import SVC
svm = SVC()

from sklearn.naive_bayes import GaussianNB
n_b = GaussianNB()


from sklearn.ensemble import VotingClassifier
vot = VotingClassifier([("LR",log_reg),
                        ("KNN", knn),
                        ("SVM", svm),
                        ("DT", dtf),
                        ("NB", n_b)])

vot.fit(X,y)
vot.score(X,y)




from sklearn.ensemble import BaggingClassifier
bag = BaggingClassifier (knn, n_estimators= 5)
bag.fit(X,y)
bag.score(X,y)

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(X,y)
