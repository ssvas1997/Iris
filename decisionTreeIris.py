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


import os
#os.environ["PATH"]= 'C:/Program Fles(x86)/Graphvz2.38/bin'
os.environ["PATH"]= 'E:\\graphviz-2.38\\release\\bin'

from sklearn.externals.six import StringIO
from IPython.display import Image
from sklearn.tree import export_graphviz
import pydotplus

dot_data= StringIO()
feat_col=['Sepal length','Sepal width','Petal length','Petal width']
export_graphviz(dectree, out_file=dot_data, filled=True, rounded=True, 
                special_characters=True, feature_names=feat_col,
                class_names=['Setosa','Versicolor','Virginica'])
graph= pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('iris.png')
Image(graph.create_png())


