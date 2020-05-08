# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 12:45:29 2020

@author: tharshi
"""


from openTSNE import TSNE
from openTSNE.callbacks import ErrorLogger

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits

import matplotlib.pyplot as plt

#%%

x, y = load_digits(return_X_y=True)
x_train, x_test, y_train, y_test = train_test_split(x, y, 
                                                    test_size=.33, 
                                                    random_state=42)

print("%d training samples" % x_train.shape[0])
print("%d test samples" % x_test.shape[0])

tsne = TSNE(
    perplexity=30,
    metric="euclidean",
    callbacks=ErrorLogger(),
    n_jobs=4,
    random_state=42,
)

embedding_train = tsne.fit(x_train)

embedding_test = embedding_train.transform(x_test)