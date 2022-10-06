#!/usr/bin/env python
# coding: utf-8

# # 16 - Representación de datos e ingeniería de características
# 
# ![](images/1.png)
# 
# ## Módulo 6 - Aprendizaje de máquina supervisado
# ### Profesor: M.Sc. Favio Vázquez

# In[2]:


from imports import *


# In[ ]:





# In[2]:


import os 
data = pd.read_csv(
    "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data", header=None, index_col=False,
    names=['age', 'workclass', 'fnlwgt', 'education',  'education-num',
           'marital-status', 'occupation', 'relationship', 'race', 'gender',
           'capital-gain', 'capital-loss', 'hours-per-week', 'native-country',
           'income'])
data = data[['age', 'workclass', 'education', 'gender', 'hours-per-week',
             'occupation', 'income']]
display(data.head(10))


# ## Datos categóricos

# ### One-Hot-Enconding (Dummy variables)

# In[3]:


data.gender.value_counts()


# In[4]:


data.workclass.value_counts()


# In[5]:


data.columns


# In[6]:


data_dummies = pd.get_dummies(data)


# In[7]:


data_dummies.columns


# In[8]:


data_dummies.head()


# In[9]:


data_dummies.dtypes


# In[10]:


features = data_dummies.loc[:, "age": "occupation_ Transport-moving"]


# In[11]:


X = features.values
y = data_dummies["income_ >50K"].values


# In[12]:


X


# In[13]:


y


# In[14]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


# In[15]:


X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=0)


# In[16]:


logreg = LogisticRegression()


# In[17]:


logreg.fit(X_train,y_train)


# In[18]:


logreg.score(X_test,y_test)


# ## Datos numéricos que codifican categorías

# In[19]:


demo_df = pd.DataFrame({"Integer Feature": [0,1,2,1],
                    "Categorical Feature": ["arroz", "pollo", "carne", "pollo"]})


# In[20]:


demo_df


# In[21]:


pd.get_dummies(demo_df)


# In[22]:


demo_df["Integer Feature"] = demo_df["Integer Feature"].astype(str)


# In[23]:


pd.get_dummies(demo_df)


# ## Castear a category

# In[24]:


demo_df = pd.DataFrame({"Integer Feature": [0,1,2,1],
                    "Categorical Feature": ["arroz", "pollo", "carne", "pollo"]})


# In[25]:


demo_df


# In[26]:


demo_df.dtypes


# In[27]:


demo_df["Integer Feature"] = demo_df["Integer Feature"].astype("category")


# In[28]:


demo_df.dtypes


# In[ ]:


demo_df


# ## Binning (discretización), modelos lineales y árboles

# In[29]:


from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor


# In[30]:


X,y = mglearn.datasets.make_wave(n_samples=100)


# In[31]:


plt.plot(X[:,0], y, "o", c="k")


# In[32]:


line = np.linspace(-3,3, 1000, endpoint=False).reshape(-1,1)


# In[33]:


dt = DecisionTreeRegressor(min_samples_split=3).fit(X,y)
lr = LinearRegression().fit(X,y)


# In[34]:


plt.plot(line, dt.predict(line), label="Árbol de decisión")
plt.plot(line, lr.predict(line), label="Regresión lineal")
plt.plot(X[:,0], y, "o", c="k")
plt.ylabel("Salida de la regresión")
plt.xlabel("Variable de entrada")
plt.legend(loc="best")
plt.show()


# In[35]:


bins = np.linspace(-3,3,11)
bins


# In[36]:


bin_data = np.digitize(X, bins=bins)


# In[37]:


X[:5]


# In[38]:


bin_data[:5]


# In[40]:


from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(sparse=False)
encoder.fit(bin_data)
X_binned = encoder.transform(bin_data)


# In[41]:


X_binned[:5]


# In[42]:


line_binned = encoder.transform(np.digitize(line, bins=bins))


# In[43]:


line_binned


# In[44]:


lr = LinearRegression().fit(X_binned,y)
dt = DecisionTreeRegressor(min_samples_split=3).fit(X_binned,y)
plt.plot(line,lr.predict(line_binned), 
         label="Regresión lineal discretizada")
plt.plot(line,dt.predict(line_binned), 
         label="Árbol de decisión discretizado", linestyle="dashed")
plt.plot(X[:,0], y, "o", c="k")
plt.vlines(bins,-3,3, linewidth=1, alpha=0.2)
plt.legend(loc="best")
plt.ylabel("Salida de la regresión")
plt.xlabel("Variable de entrada")


# ### Características polimoniales e interacciones

# El binning nos permite expandir una variable continua. Otra forma de hacerlo es usar polinomios de las variables originales. Para una característica x, qusieramos considerar $x^2$, $x^3$, $x^4$, etc. Esto viene implementado en la función PolynomialFeatures del módulo de preprocesado de scikit-learn.

# In[45]:


X[:4]


# In[67]:


from sklearn.preprocessing import PolynomialFeatures

# Crear un polinomio de rango 10
# Eliminaremos la opción include_bias porque agrega una característica
# que siempre es igual 1
poly = PolynomialFeatures(degree=10, include_bias=False)
poly.fit(X)
X_poly = poly.transform(X)


# In[68]:


X.shape


# In[69]:


X_poly.shape


# In[70]:


X[:5]


# In[71]:


X_poly[:5]


# In[72]:


poly.get_feature_names()


# In[73]:


lr = LinearRegression().fit(X_poly,y)
line_poly = poly.transform(line)
plt.plot(line, lr.predict(line_poly), label="Regresión lineal polinomial")
plt.plot(X[:,0], y, "o", c="k")
plt.legend(loc="best")
plt.ylabel("Salida de la regresión")
plt.xlabel("Variable de entrada")


# In[74]:


from sklearn.svm import SVR

for gamma in [1,10]:
    svr = SVR(gamma=gamma).fit(X,y)
    plt.plot(line, svr.predict(line), label="SVR gamma={}".format(gamma))
plt.plot(X[:,0], y, "o", c="k")
plt.legend(loc="best")
plt.ylabel("Salida de la regresión")
plt.xlabel("Variable de entrada")


# ## Ejemplo

# In[75]:


from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


# In[76]:


boston = load_boston()

# Método de la exclusión
X_train, X_test, y_train, y_test = train_test_split(boston.data,
                                                    boston.target,
                                                    random_state=0)


# In[77]:


# reescalamimento

from sklearn.preprocessing import MinMaxScaler
data = [[-1, 2], [-0.5, 6], [0, 10], [1, 18]]
scaler = MinMaxScaler()
print(scaler.fit(data))
print("\n")
print(scaler.data_max_)
print("\n")
print(scaler.transform(data))


# In[78]:


# reescalar los datos
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[79]:


# Encontrar características polimoniales e interacciones hasta grado 2
poly = PolynomialFeatures(degree=2).fit(X_train_scaled)
X_train_poly = poly.transform(X_train_scaled)
X_test_poly = poly.transform(X_test_scaled)


# In[80]:


X_train.shape


# In[81]:


X_train_poly.shape


# In[82]:


poly.get_feature_names()


# In[83]:


from sklearn.linear_model import Ridge
ridge = Ridge().fit(X_train_scaled,y_train)
print("Score sin interacciones {:.3f}".format(ridge.score(X_test_scaled,
                                                         y_test)))
ridge = Ridge().fit(X_train_poly,y_train)
print("Score con interacciones {:.3f}".format(ridge.score(X_test_poly,
                                                         y_test)))


# In[84]:


from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=100).fit(X_train_scaled,y_train)
print("Score sin interacciones {:.3f}".format(rf.score(X_test_scaled,
                                                         y_test)))
rf = RandomForestRegressor(n_estimators=100).fit(X_train_poly,y_train)
print("Score con interacciones {:.3f}".format(rf.score(X_test_poly,
                                                         y_test)))


# ## Transformaciones no lineales univariadas

# La mayoría de los modelos funcionan mejor cuando cuando cada característica (y en regresión también el target) tienen una distribución cercana a la Gaussiana. Las transformaciones como el logaritmo o el exponencial nos permite transformar las distribuciones de las variables para que tengan una forma gaussiana.

# In[85]:


rnd = np.random.RandomState(0)
X_org = rnd.normal(size=(1000,3))
w = rnd.normal(size=3)


# In[86]:


X_org


# In[87]:


w


# In[88]:


X = rnd.poisson(10*np.exp(X_org))
y = np.dot(X_org,w)


# In[89]:


X[:3]


# In[90]:


y[:3]


# In[ ]:


np.bincount(X[:,0])


# In[91]:


bins = np.bincount(X[:,0])
plt.bar(range(len(bins)), bins, color="b")
plt.ylabel("Número de apariciones")
plt.xlabel("Valor")


# In[92]:


from sklearn.linear_model import Ridge
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=0)
score = Ridge().fit(X_train,y_train).score(X_test,y_test)
print("Score del test: {:.3f}".format(score))


# In[93]:


X_train_log = np.log(X_train + 1)
X_test_log = np.log(X_test + 1)


# In[94]:


plt.hist(X_train_log[:,0], bins=25, color="gray")
plt.ylabel("Número de apariciones")
plt.xlabel("Valor")


# In[95]:


score = Ridge().fit(X_train_log,y_train).score(X_test_log,y_test)
print("Score del test: {:.3f}".format(score))


# Encontrar la transformación que funcione mejor para un dataset y un modelo no es obvio, parece hasta un arte. 
