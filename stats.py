#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from pylab import rcParams
import seaborn as sns
from collections import Counter

tr_path = 'data/diabetes.csv'
df = pd.read_csv(tr_path)


# In[2]:


df.head(5)


# In[3]:


df.info()


# In[4]:


df.isnull().sum()


# In[5]:


k = 9 #number of variables for heatmap
cols = df.corr().nlargest(k, 'Outcome')['Outcome'].index
cm = df[cols].corr()
plt.figure(figsize=(10,6))
sns.heatmap(cm, annot=True, cmap = 'viridis')


# In[190]:


df.describe([0.05,0.25,0.50,0.75,0.90,0.95,0.99]).T


# In[135]:


# No hay datos vacíos, sin embargo se observa que el minimo en todas las columnas es cero, lo cual sólo tiene sentido para
# la columna que registra el número de embarazos
df.describe()


# In[136]:


df.columns


# # EDA

# In[8]:


top_age = df.Age.value_counts().head(15)
top_age

plt.figure(figsize=(12,6))
plt.xticks(rotation=75)
sns.barplot(x=top_age.index, y=top_age)


# In[138]:


def plot_hist(df,feature):
    plt.hist(df[feature], bins = 50)
    plt.xlabel(feature)
    plt.ylabel("Frequency")
    plt.title("{} distribution with hist".format(feature))
    plt.show()


# In[140]:


numericVar = df.columns[:-1]
for n in numericVar:
    plot_hist(df,n)


# In[162]:


def ZEROs(df):
    print("               # of ZEROs  \t Length \t  Percent "  )
    print("------------------------------------------------------"  )

    for i in range(6) :
        feature=df[df.columns[i]]
        ZeroSum=(feature==0).sum()
        Percent=int(round((ZeroSum/(len(feature)))*100))
        print(df.columns[i], "\t:", ZeroSum , "\t\t:", len(feature), "\t \t:", Percent,"% "  )


# In[163]:


ZEROs(df)


# In[ ]:





# In[164]:


# Defining X and y
X = df.iloc[:,:-1]   # Dependants
y = df.iloc[:, -1]   # Outcome


# In[166]:


from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=0, strategy='median')

X =imputer.fit_transform(X)    # it transform it as an array
X = pd.DataFrame(X,columns=df.columns[:-1])     # Needs to be made as DataFrame again
df=pd.concat([X,y],axis=1)
df2=df  # ıt xill be used later in KNN 


# In[167]:


ZEROs(df)


# In[ ]:





# In[168]:


# para eliminar los outliers

z_scores = stats.zscore(data)
abs_z_scores = np.abs(z_scores)
entradas_filtradas = (abs_z_scores < 3).all(axis=1)
 
data_nuevo = data[entradas_filtradas]


# In[169]:


numericVar = data.columns[:-1]
for n in numericVar:
    plot_hist(data_nuevo,n)


# In[170]:


# no se han eliminado todos los outliers, sobre todo en las variables insuline y SkinThickness
def detect_outliers(df, features):
    outlier_indices = []
    
    for c in features:
        # 1st quartile Q1
        Q1 = np.percentile(df[c], 25)
        # 3st quartile Q3
        Q3 = np.percentile(df[c], 75)
        # IQR
        IQR = Q3 - Q1
        # Outlier step
        outlier_step = IQR * 1.5
        # detect outlier and their indices
        outlier_list_col = df[(df[c] < Q1 - outlier_step) | (df[c] > Q3 + outlier_step)].index
        # store indices
        outlier_indices.extend(outlier_list_col)
        
    outlier_indices = Counter(outlier_indices)
    multiple_outliers = list(i for i, v in outlier_indices.items() if v > 2)
    
    return multiple_outliers


# In[172]:


df.loc[detect_outliers(df, ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
       'BMI', 'DiabetesPedigreeFunction', 'Age'])]


# In[173]:


df = df.drop(detect_outliers(df, ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
       'BMI', 'DiabetesPedigreeFunction', 'Age']), axis = 0).reset_index(drop= True)
df.describe()


# In[174]:


sns.heatmap(data.corr(), annot = True, cmap='Blues', fmt = ".2f")
plt.show()


# In[ ]:





# In[175]:


x=X
for i in x:
    g = sns.distplot(x[i], color = "b", label = "Skewness : %.2f"%(x[i].skew()))
    g = g.legend(loc = "best")
    plt.show()


# In[176]:


for i in X:
    g = sns.FacetGrid(df, col = "Outcome")
    g.map(sns.distplot, i, bins= 25)        
    plt.show()


# In[177]:


def plotHistogram(values,label,feature,title):
    sns.set_style("whitegrid")
    plotOne = sns.FacetGrid(values, hue=label,aspect=2)
    plotOne.map(sns.distplot,feature,kde=False)
    plotOne.set(xlim=(0, values[feature].max()))
    plotOne.add_legend()
    plotOne.set_axis_labels(feature, 'Proportion')
    plotOne.fig.suptitle(title)
    plt.show()
for i in X:
    plotHistogram(df,"Outcome",i,' Diagnosis (Blue = Healthy; Orange = Diabetes)')


# In[178]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)


# In[179]:


print("X_train", X_train.shape)
print("X_test", X_test.shape)
print("y_train", y_train.shape)
print("y_test", y_test.shape)


# In[180]:


from sklearn.preprocessing import StandardScaler
scale = StandardScaler()


# In[181]:


X_train = scale.fit_transform(X_train)
X_test = scale.transform(X_test)


# In[182]:


from sklearn.linear_model import LogisticRegression
log_model = LogisticRegression()


# In[183]:


log_model.fit(X_train, y_train)
y_pred = log_model.predict(X_test)
y_train_pred = log_model.predict(X_train)


# In[184]:


print("log_model.coef_:",log_model.coef_,"\nlog_model.intercept_ :",log_model.intercept_)


# In[185]:


from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score


# In[ ]:




