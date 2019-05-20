# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#prediction of gender 
from sklearn import tree

x=[[189,65,12],[165,56,8],[189,615,112],[165,156,81],[189,165,121],[165,156,18]]
y=['female','female','male','female','male','female']

clf=tree.DecisionTreeClassifier()
clf=clf.fit(x,y)

prediction=clf.predict([[190,70,43]])
print (prediction)