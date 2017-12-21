#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 17 15:28:35 2017

@author: shubhi
"""

from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
import numpy as np


iris = datasets.load_iris()
classifier = KNeighborsClassifier()

classifier.fit( iris.data , iris.target)

error=np.mean((classifier.predict(iris.data) - iris.target)**2)

print (error)
