# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

df= pd.read_csv("E:\Srinivas\iris.csv")
df.dtypes
df.describe()
df['Petal width'].plot.hist()
sns.pairplot(df,hue='Class')
features=df[['Sepal length','Sepal width','Petal length','Petal width']].values
classes=df['Class'].values
features.shape
(train_feat,test_feat,train_classes,test_classes)= train_test_split(features,classes,train_size=0.7,random_state=1)
#Training
dectree= DecisionTreeClassifier()
dectree.fit(train_feat,train_classes) #Supervised Learning
#Testing
pred= dectree.predict(test_feat)
print("Accuracy:",metrics.accuracy_score(test_classes,pred))
#Predicting a single input feature
sepl= input("Sepal length")
sepw= input("Sepal width")
petl= input("Petal length")
petw= input("Petal width")
print(sepl,sepw,petl,petw)
pr=dectree.predict(np.column_stack([sepl,sepw,petl,petw]))
print("Predicted Species is:",pr)
