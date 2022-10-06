#!/usr/bin/env python
# coding: utf-8

# # 17 -Ingeniería de características (Selección automática de características)
# 
# ![](images/1.png)
# 
# ## Módulo 6 - Aprendizaje de máquina supervisado
# ### Profesor: M.Sc. Favio Vázquez

# In[5]:


from imports import *


# Agregar más características aumenta la complejidad de los algoritmos, y también aumenta la posibilidad del sobreajuste. 
# 
# Cuando agregamos nuevas características, no todas serán relevantes para la predicción, por lo tanto es una buena idea reducir el número de variables a las más importantes, y descartar el resto. Comúnmente esto nos lleva a modelos más simples y generalizables.

# ## Estadísticas univariadas

# - Validar si existe una relación estadísticamente significante entre cada característica y el target.
# - Se escogen las variables que den más confianza de relación. 
# - Esto se llama Analysis of Variance (ANOVA).
# - Se va haciendo variable por variable individualmente.

# In[1]:


from sklearn.datasets import load_breast_cancer
from sklearn.feature_selection import SelectPercentile
from sklearn.model_selection import train_test_split


# In[2]:


cancer = load_breast_cancer()


# In[3]:


len(cancer.feature_names)


# In[6]:


# Agregar ruido a los datos para hacer más compleja la información
# Configurar mi semilla
rng = np.random.RandomState(42)
noise = rng.normal(size=(len(cancer.data), 50))
# Agegar el ruido, las primeras 30 features son las originales
# Las siguientes 50 features son ruido
X_w_noise = np.hstack([cancer.data,noise])


# In[7]:


# Método de la exclusión
X_train, X_test, y_train, y_test = train_test_split(X_w_noise, 
                                                    cancer.target,
                                                    random_state=0)


# In[8]:


# Usar f_classif y SelectPercentile para escoger el 10% de las variables
select = SelectPercentile(percentile=10)
select.fit(X_train,y_train)
# Transformar el dataset de entrenamiento
X_train_selected = select.transform(X_train)


# In[9]:


X_train.shape


# In[10]:


X_train_selected.shape


# In[11]:


# Máscara para saber qué variables escogió
mask = select.get_support()
print(mask)

# Visualizar la máscara -- negro es verdad, blanco es falso
plt.matshow(mask.reshape(1,-1), cmap="gray_r")
plt.xlabel("Sample")
plt.yticks(())


# In[12]:


from sklearn.linear_model import LogisticRegression

X_test_selected = select.transform(X_test)

lr = LogisticRegression()
lr.fit(X_train,y_train)
print("Score con todos los datos: {:.3f}".format(lr.score(X_test,y_test)))
lr.fit(X_train_selected,y_train)
print("Score con datos seleccionados: {:.3f}".format(lr.score(X_test_selected,y_test)))


# - Selección de características basadas en un modelo
# - Selección de características iterativamente

# ## Selección basada en modelos

# In[13]:


from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier


# In[14]:


cancer = load_breast_cancer()


# In[15]:


# Método de la exclusión
X_train, X_test, y_train, y_test = train_test_split(cancer.data, 
                                                    cancer.target,
                                                    random_state=0)


# In[16]:


select = 2(RandomForestClassifier(random_state=42), 
                     threshold="median")


# In[17]:


select.fit(X_train, y_train)


# In[18]:


X_train_best = select.transform(X_train)


# In[19]:


X_train.shape


# In[20]:


X_train_best.shape


# In[ ]:


mask = select.get_support()
plt.matshow(mask.reshape(1,-1), cmap="gray_r")
plt.xlabel("Sample")
plt.yticks(())


# In[21]:


# Modelo para predecir
from sklearn.linear_model import LogisticRegression


# In[22]:


X_test_best = select.transform(X_test)


# In[23]:


clf = LogisticRegression().fit(X_train_best,y_train)


# In[24]:


clf.score(X_test_best,y_test)


# ## Con ruido

# In[27]:


# Método de la exclusión
X_train, X_test, y_train, y_test = train_test_split(X_w_noise, 
                                                    cancer.target,
                                                    random_state=0)


# In[40]:


select_noise = SelectFromModel(RandomForestClassifier(random_state=42), 
                         threshold="median")


# In[41]:


select_noise.fit(X_train, y_train)


# In[42]:


mask = select_noise.get_support()
plt.matshow(mask.reshape(1,-1), cmap="gray_r")
plt.xlabel("Sample")
plt.yticks(())


# In[43]:


X_train_best = select_noise.transform(X_train)


# In[44]:


X_test_best = select_noise.transform(X_test)


# In[45]:


clf = LogisticRegression().fit(X_train_best,y_train)


# In[46]:


clf.score(X_test_best,y_test)


# ## Selección de características iterativamente

# In[47]:


# Método de la exclusión
X_train, X_test, y_train, y_test = train_test_split(cancer.data, 
                                                    cancer.target,
                                                    random_state=0)


# In[48]:


# Recursive Feature Elimination (RFE)
from sklearn.feature_selection import RFE

select = RFE(RandomForestClassifier(random_state=42), 
             n_features_to_select=10)


# In[49]:


select.fit(X_train,y_train)


# In[50]:


mask = select.get_support()
plt.matshow(mask.reshape(1,-1), cmap="gray_r")
plt.xlabel("Sample")
plt.yticks(())


# In[54]:


for i in range(X_train.shape[1]):
    print("Id: {}, Seleccionada: {}".format(i,select.support_[i]))


# In[55]:


for idx,name in enumerate(cancer.feature_names):
    print(idx,name)


# In[51]:


X_train_rfe = select.transform(X_train)
X_test_rfe = select.transform(X_test)


# In[52]:


clf = LogisticRegression().fit(X_train_rfe,y_train)


# In[53]:


clf.score(X_test_rfe,y_test)

