#!/usr/bin/env python
# coding: utf-8

# # 14 - SVM para clasificación
# 
# ![](images/portada_nb_FAV.png)
# 
# ## Módulo 6 - Aprendizaje de máquina supervisado
# ### Profesor: M.Sc. Favio Vázquez

# In[ ]:


from imports import *


# ## Support Vector Machines
# 
# 
# EL SVM es un tipo de técnica ML que se puede utilizar tanto para clasificación como para regresión. Hay dos tipos principales para soportar problemas lineales y no lineales. Linear SVM no tiene kernel y encuentra una solución lineal al problema con un margen mínimo. El SVM con Kernel se utiliza cuando la solución no se puede separar linealmente.
# 
# 
# ## Teoría básica
# 
# 
# Un hiperplano se deriva del modelo que maximiza el margen de clasificación. Si N características están presentes, el hiperplano será un subespacio dimensional N-1. Los nodos de frontera en el espacio de características se denominan vectores de soporte. Según su posición relativa, se deriva el margen máximo y se dibuja un hiperplano óptimo en el punto medio.
# 
# ![](https://static.javatpoint.com/tutorial/machine-learning/images/support-vector-machine-algorithm.png)
# 
# Es un problema de optimizacion,
# 
# ![](https://i.ibb.co/TgrvCft/optimizasyon.jpg)
# 
# La optimización anterior funciona bien para soluciones separables completamente lineales. Para manejar valores atípicos, necesitamos un término flexible como se muestra a continuación. El segundo término usa la pérdida de bisagra para obtener una variable de holgura.
# 
# ![](https://i.ibb.co/ZKmqy58/hinge.jpg)
# 
# C es el parámetro de regularidad que equilibra la penalización perdida y el ancho del margen. 
# 
# La lógica básica es que para minimizar la función de costo, w se ve obligado a ajustar con el máximo margen entre clases. El valor de C decidirá el nivel de regularidad aplicado a los conjuntos de datos. Decide el margen de nivel (suave / duro) que se aplicará a los conjuntos de datos. En resumen, C es el nivel de ignorancia hacia los valores atípicos.
# 
# Cuando el conjunto de datos no se puede separar linealmente, se utiliza una función de kernel para derivar un nuevo subplano para todos los datos de entrenamiento. La distribución de las etiquetas en el nuevo subplano será tal que los datos de entrenamiento sean linealmente separables. A continuación, una curva lineal clasificará las etiquetas en el plano inferior. Cuando los resultados de la clasificación se reflejan en el espacio de características, obtenemos una solución no lineal.
# 
# ![](https://i.ibb.co/n3NbrxR/Ek-A-klama-2020-08-29-113927.jpg)
# 
# El único cambio en la ecuación aquí es describir una nueva función del núcleo. La nueva ecuación se verá así:
# 
# ![](https://i.ibb.co/X28y1vG/Ek-A-klama-2020-08-29-114044.jpg)
# 
# $X_i$ será reemplazado por $\phi(x_i)$, que transformará el conjunto de datos en el nuevo hiperplano.

# ## El truco del kernel

# Nos permite aprender un clasificador en más dimensiones sin calcular nuevas variables. Lo que hace es calcular la distancia (calcula el producto escalar) de los datos para la representación extendida, sin en realidad hacer la extensión.
# 
# Algunos kernels famosos:
# 
# - Polinómico
# - Radial basis function (RBF) --> Kernel Gaussiano.

# Durante el entrenamiento, los SVM aprenden la importancia de los datos de entrenamiento para representar la frontera entre las clases. Típicamente solo un subconjunto de los puntos importan para definir la frontera: los que caen el borde entre las clases. A éstos los llamamos vectores de sorporte, y de ahí viene el nombre del algoritmo. 

# La distancia entre los puntos en el kernel gaussiano es :
# 
# $$
# k_{rbf}(x_1,x_2) = \exp(-\gamma ||x_1 - x_2||^2)
# $$

# Donde $||x_1 - x_2||$ es la distancia euclideana y $\gamma$ es un parámetro que controla la anchura del kernel.

# ### Ejemplo

# In[ ]:


X,y = mglearn.tools.make_handcrafted_dataset()


# In[ ]:


mglearn.discrete_scatter(X[:, 0], X[:, 1], y)


# In[ ]:


from sklearn.svm import SVC

svm = SVC(kernel="rbf", C=10, gamma=0.1).fit(X,y)


# In[ ]:


mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
mglearn.plots.plot_2d_separator(svm, X, eps=0.5)


# In[ ]:


svm.support_vectors_


# In[ ]:


mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
mglearn.plots.plot_2d_separator(svm, X, eps=0.5)

sv = svm.support_vectors_
sv_labels = svm.dual_coef_.ravel() > 0
mglearn.discrete_scatter(sv[:,0], sv[:,1], sv_labels, s=15, 
                         markeredgewidth=3)
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")


# In[ ]:


fig, axes = plt.subplots(3,3, figsize=(15,10))

for ax, C in zip(axes, [-1, 0, 3]):
    for a, gamma in zip(ax, range(-1,2)):
        mglearn.plots.plot_svm(log_C=C, log_gamma=gamma, ax=a)
        
axes[0,0].legend(["clase 0", "clase 1", "sv clase 0", "sv clase 1"],
                ncol=4, loc=(0.9,1.2))


# ### Ventajas
# - SVM usa el número de kernel para resolver soluciones complejas.
# - SVM utiliza una función de optimización convexa cuyo mínimo global siempre es alcanzable.
# - La pérdida de bisagra proporciona una mayor precisión.
# - Los valores atípicos se pueden manejar bien utilizando la constante de margen suave C.
# 
# 
# ### Desventajas
# - La pérdida de bisagra conduce a escasez.
# - Los hiperparámetros y núcleos deben ajustarse cuidadosamente para una precisión adecuada.
# - Mayor tiempo de entrenamiento para conjuntos de datos más grandes.
# 
# 
# ### Hiperparámetros
# - **Constante de margen suave (C):**
#     - Es un hiperparámetro que decide el nivel de penalización de los valores atípicos. Es el inverso del parámetro de regularización. Cuando C es grande, los valores atípicos recibirán una penalización alta y se creará un margen rígido. Cuando C es pequeño, los valores atípicos se ignoran y el margen es grande.
# - **El grado del polinomio en el núcleo polinómico (d):**
#     - Cuando d = 1, es equivalente a un núcleo lineal. Cuando D es más alto, el núcleo es lo suficientemente flexible como para distinguir patrones complejos proyectándolos en un nuevo hiperplano.
# 
# - **Parámetro de ancho (γ) en el kernel gaussiano:**
#     - Gamma decide el ancho de la curva gaussiana. El ancho aumenta con el aumento de gamma.

# In[1]:


# Import the necessary packages

import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import sklearn
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import f1_score, recall_score, precision_score, confusion_matrix
from sklearn.metrics import r2_score, roc_auc_score, roc_curve, classification_report
from sklearn.svm import SVC 

import warnings
warnings.filterwarnings('ignore')


# In[2]:


# Import and read dataset

data = pd.read_csv("data/heart_failure_clinical_records_dataset.csv")

data.head(10)


# In[ ]:


data.describe()


# In[ ]:


data.shape


# In[ ]:


data.columns


# In[ ]:


data["DEATH_EVENT"].hist()


# In[5]:


inp_data = data.drop(data[['DEATH_EVENT']], axis=1)
out_data = data[['DEATH_EVENT']]

scaler = StandardScaler()
inp_data = scaler.fit_transform(inp_data)

X_train, X_test, y_train, y_test = train_test_split(inp_data, out_data, test_size=0.2, random_state=42)


# In[6]:


print("X_train Shape : ", X_train.shape)
print("X_test Shape  : ", X_test.shape)
print("y_train Shape : ", y_train.shape)
print("y_test Shape  : ", y_test.shape)


# In[ ]:


# basic method
clf = SVC() 
clf.fit(X_train, y_train) 

y_pred = clf.predict(X_test)

print('Accuracy Score: {:.4f}'.format(accuracy_score(y_test, y_pred)))
print('SVC f1-score  : {:.4f}'.format(f1_score(y_pred, y_test)))
print('SVC precision : {:.4f}'.format(precision_score(y_pred, y_test)))
print('SVC recall    : {:.4f}'.format(recall_score(y_pred, y_test)))
print("\n",classification_report(y_pred, y_test))


# In[ ]:


from sklearn.metrics import confusion_matrix

confusion_matrix(y_test,y_pred)


# In[ ]:


# find best parameters with SVC | Step 1
kernels = list(['linear', 'rbf', 'poly', 'sigmoid'])
c = list([0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000])
gammas = list([0.1, 1, 10, 100])

clf = SVC()
param_grid = dict(kernel=kernels, C=c, gamma=gammas)
grid = GridSearchCV(clf, param_grid, cv=10, n_jobs=-1, verbose=1)
grid.fit(X_train, y_train.values.ravel())
grid.best_params_


# In[ ]:


clf = SVC(C=10, gamma=0.1, kernel='linear') 
clf.fit(X_train, y_train) 

y_pred = clf.predict(X_test)

print('Accuracy Score: {:.4f}'.format(accuracy_score(y_test, y_pred)))
print('SVC f1-score  : {:.4f}'.format(f1_score(y_pred, y_test)))
print('SVC precision : {:.4f}'.format(precision_score(y_pred, y_test)))
print('SVC recall    : {:.4f}'.format(recall_score(y_pred, y_test)))
print("\n",classification_report(y_pred, y_test))


# ## Con SMOTE

# In[ ]:


get_ipython().system('pip install imblearn')


# In[7]:


from imblearn.over_sampling import SMOTE

sms = SMOTE(random_state=12345)
X_res, y_res = sms.fit_resample(inp_data, out_data)
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2,
                                                    stratify=y_res, random_state=42)

print("X_train Shape : ", X_train.shape)
print("X_test Shape  : ", X_test.shape)
print("y_train Shape : ", y_train.shape)
print("y_test Shape  : ", y_test.shape)


# In[11]:


# find best parameters with SVC | Step 2
kernels = list(['linear', 'rbf', 'poly', 'sigmoid'])
c = list([0.001, 0.01, 0.1, 1, 10, 100])
gammas = list([0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])

clf = SVC()
param_grid = dict(kernel=kernels, C=c, gamma=gammas)
grid = GridSearchCV(clf, param_grid, cv=10, n_jobs=-1)
grid.fit(X_train, y_train.values.ravel())
grid.best_params_


# In[16]:


clf = SVC(C=10, gamma=0.2, kernel='rbf') 
clf.fit(X_train, y_train) 

y_pred = clf.predict(X_test)

print('Accuracy Score: {:.4f}'.format(accuracy_score(y_test, y_pred)))
print('SVC f1-score  : {:.4f}'.format(f1_score(y_pred, y_test)))
print('SVC precision : {:.4f}'.format(precision_score(y_pred, y_test)))
print('SVC recall    : {:.4f}'.format(recall_score(y_pred, y_test)))
print("\n",classification_report(y_pred, y_test))


# In[13]:


y_pred = clf.predict(X_test)

cf_matrix = confusion_matrix(y_pred, y_test)
sns.heatmap((cf_matrix / np.sum(cf_matrix)*100), annot = True, fmt=".2f", cmap="Blues")


# In[14]:


# sanity check
scores = [] 
for i in range(0,1000): 
    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2)
    clf = SVC(kernel='rbf', C=10, gamma=0.2) 
    clf.fit(X_train, y_train)
    scores.append(accuracy_score(clf.predict(X_test), y_test)) 

plt.hist(scores)
plt.show()

