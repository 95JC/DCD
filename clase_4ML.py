#!/usr/bin/env python
# coding: utf-8

# KNN es uno de los algoritmos de ML más simples. La idea detrás de este algoritmo es encontrar los puntos más cercanos a un valor ("sus vecinos más cercanos") para poder encontrar cual sería el valor de este punto o a qué clase pertenece.

# In[2]:


import pandas as pd
import mglearn
import matplotlib.pyplot as plt
import numpy as np


# In[3]:


from imports import *


# ## KNN para regresión

# In[6]:


X, y = mglearn.datasets.make_wave(n_samples = 40)
plt.plot(X, y, 'o')
plt.ylim(-3, 3)
plt.xlabel('feature')
plt.ylabel('target')
plt.show()


# In[7]:


mglearn.plots.plot_knn_regression(n_neighbors = 1)


# In[8]:


mglearn.plots.plot_knn_regression(n_neighbors = 3)


# In[9]:


# Ejemplo


# In[10]:


from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split

X, y = mglearn.datasets.make_wave(n_samples = 40)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)


# In[11]:


reg = KNeighborsRegressor(n_neighbors = 3)
reg.fit(X_train, y_train)


# In[12]:


y_test


# In[13]:


reg.predict(X_test)


# In[14]:


reg.score(X_test, y_test) # r^2


# In[15]:


fig, axes = plt.subplots(1,3, figsize=(15,4))

line = np.linspace(-3,3,1000).reshape(-1,1)

for n_neighbors, ax in zip([1,3,9], axes):
    reg = KNeighborsRegressor(n_neighbors)
    reg.fit(X_train,y_train)
    ax.plot(line, reg.predict(line))
    ax.plot(X_train, y_train, "^", c=mglearn.cm2(0), markersize=8)
    ax.plot(X_test, y_test, "v", c=mglearn.cm2(1), markersize=8)
    ax.set_title("{} vecinos \n score train: {:.2f} score test: {:.2f}".format(
    n_neighbors, reg.score(X_train,y_train), reg.score(X_test,y_test)))
    ax.set_xlabel("Característica")
    ax.set_label("Target")
    axes[0].legend(["Predicciones del modelo", "Datos de entrenamiento/target", 
                   "Datos de prueba/target"], loc="best")


# In[ ]:




