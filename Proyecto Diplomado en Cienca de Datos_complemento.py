#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import date
import seaborn as sns
import datetime
from datetime import timedelta
from datetime import datetime
from sklearn.linear_model import LinearRegression # for building a linear regression model
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR # for building support vector re
import math
from tqdm import tqdm


# In[ ]:





# In[2]:


pip install statsmodels


# In[3]:


pip install skforecast


# In[4]:


import numpy as np
import pandas as pd

# Gráficos
# ==============================================================================
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
plt.rcParams['lines.linewidth'] = 1.5
get_ipython().run_line_magic('matplotlib', 'inline')

# Modelado y Forecasting
# ==============================================================================
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from skforecast.ForecasterAutoreg import ForecasterAutoreg
from skforecast.ForecasterAutoregCustom import ForecasterAutoregCustom
from skforecast.ForecasterAutoregMultiOutput import ForecasterAutoregMultiOutput
from skforecast.model_selection import grid_search_forecaster
from skforecast.model_selection import backtesting_forecaster

from joblib import dump, load

# Configuración warnings
# ==============================================================================
import warnings
# warnings.filterwarnings('ignore')


# In[5]:


fiscal = pd.read_excel('balanza_fiscal.xlsx') 
fx = pd.read_excel('tipo_de_cambio.xlsx') 
igae = pd.read_excel('IGAE.xlsx') 
petroleo = pd.read_excel('balanza_petroleo.xlsx') 
deuda = pd.read_excel('deuda.xlsx') 
inflacion = pd.read_excel('inflacion.xlsx') 
remesas = pd.read_excel('remesas.xlsx') 
M = pd.read_excel('oferta_monetaria.xlsx') 
pib = pd.read_excel('PIB_desestacionalizado.xlsx') 
oa_da = pd.read_excel('OA_DA.xlsx')
inversion = pd.read_excel('inversion.xlsx')


# In[6]:


fechas_fiscal = fiscal['Título'].iloc[8:].tolist()
fechas_fiscal = [fecha.date() for fecha in fechas_fiscal]
fechas_fiscal = pd.Series(fechas_fiscal)


# In[7]:


gasto = fiscal['Gastos Presupuestales del Sector Público, Clasificación Económica, Gasto presupuestario'].iloc[8:].tolist()
impuestos = fiscal['Ingresos y Gastos Presupuestales del Gobierno Federal, Medición por Ingreso-Gasto, Flujos de Caja, Ingreso total, Tributarios'].iloc[8:].tolist()
gasto = pd.Series(gasto)
impuestos = pd.Series(impuestos)

b_fiscal = pd.concat([fechas_fiscal, gasto, impuestos], axis = 1)
b_fiscal.columns = ['fecha', 'Gasto del gobierno)(mdp)', 'ingresos por impuestos(mdp)']
b_fiscal


# In[8]:


fx.head(12)


# In[9]:


fechas_fx = fx['Título'].iloc[8:]
fechas_fx = [fecha.date() for fecha in fechas_fx]
fechas_fx = pd.Series(fechas_fx)
fechas_fx


# In[10]:


valor_fx = fx['Serie histórica del tipo de cambio, Tipo de cambio peso dólar desde 1954'].iloc[8:].tolist()
valor_fx = pd.Series(valor_fx)
valor_fx


# In[11]:


FX = pd.concat([fechas_fx, valor_fx], axis = 1)
FX.columns = ['fecha', 'tipo_de_cambio']
FX


# In[12]:


igae.columns


# In[13]:


fechas_igae = igae['Título'].iloc[8:].tolist()
fechas_igae = [fecha.date() for fecha in fechas_igae]
fechas_igae = pd.Series(fechas_igae)
fechas_igae


# In[14]:


datos_igae = igae.iloc[:, 1].iloc[8:].tolist()
datos_igae = pd.Series(datos_igae)
datos_igae


# In[15]:


indicador_act_econ = pd.concat([fechas_igae, datos_igae], axis = 1)
indicador_act_econ.columns = ['fecha', 'igae']
indicador_act_econ


# In[16]:


fechas_petroleo = petroleo['Título'].iloc[8:].tolist()
fechas_petroleo = [fecha.date() for fecha in fechas_petroleo]
fechas_petroleo = pd.Series(fechas_petroleo)
fechas_petroleo


# In[17]:


exportaciones_petroleras = petroleo['Balanza comercial de mercancías de México, Exportaciones totales, Petroleras'].iloc[8:].tolist()
exportaciones_petroleras = pd.Series(exportaciones_petroleras)
exportaciones_petroleras


# In[18]:


crudo = pd.concat([fechas_petroleo, exportaciones_petroleras], axis = 1)
crudo.columns = ['fecha', 'exportaciones_petroleo (miles de dolares)']
crudo


# In[19]:


deuda.head()


# In[20]:


fechas_deuda = deuda['Banco de México'].iloc[9:].tolist()
fechas_deuda = [fecha.date() for fecha in fechas_deuda]
fechas_deuda = pd.Series(fechas_deuda)
fechas_deuda


# In[21]:


datos_deuda = deuda['Unnamed: 1']
datos_deuda = datos_deuda.iloc[9:].tolist()
deuda_neta = pd.Series(datos_deuda)
deuda_neta_mdp = deuda_neta*1000
deuda_neta_mdp


# In[22]:


debt = pd.concat([fechas_deuda, deuda_neta_mdp], axis = 1)
debt.columns = ['fecha', 'deuda (mdp)']
debt


# In[23]:


inflacion.head(12)


# In[24]:


fechas_inflacion = inflacion['Título'].iloc[8:].tolist()
fechas_inflacion = [fecha.date() for fecha in fechas_inflacion]
fechas_inflacion = pd.Series(fechas_inflacion)
fechas_inflacion


# In[25]:


dato_inflacion = inflacion['Índice Nacional de Precios al consumidor, Variación mensual'].iloc[8:].tolist()
dato_inflacion = pd.Series(dato_inflacion)
dato_inflacion


# In[26]:


pi = pd.concat([fechas_inflacion, dato_inflacion], axis = 1)
pi.columns = ['fecha', 'inflacion']
pi


# In[27]:


fechas_n = []
for i in range(len(pi['fecha'].tolist()) - 1):
    inicio = pi['fecha'].tolist()[i]
    fin = pi['fecha'].tolist()[i + 1]
    lista_fechas = [inicio + timedelta(days = d) for d in range((fin - inicio).days)] 
    fechas_n.append(lista_fechas)
fechas_n

ultimo_mes = [pi['fecha'].tolist()[-1] + timedelta(days = d) for d in range(30)]
fechas_n.append(ultimo_mes)


# In[28]:


datos_pi = []
for i in range(len(pi['inflacion'].tolist())):
    lista_datos = np.full(len(fechas_n[i]), pi['inflacion'].tolist()[i])
    lista_datos = lista_datos.tolist()
    datos_pi.append(lista_datos)


# In[29]:


pi_dia = []
fechas_pi = []
for k in range(len(datos_pi)):
    for j in range(len(datos_pi[k])):
        pi_dia.append(datos_pi[k][j])
        fechas_pi.append(fechas_n[k][j])

pi_dia = pd.Series(pi_dia)
fechas_pi = pd.Series(fechas_pi)

pi_df = pd.concat([fechas_pi, pi_dia], axis = 1)
pi_df.columns = ['fecha', 'inflacion']

pi_df


# In[30]:


fechas_agregado = oa_da['Título'].iloc[8:].tolist()
fechas_agregado = [fecha.date() for fecha in fechas_agregado]
fechas_agregado = pd.Series(fechas_agregado)
fechas_agregado


# In[31]:


consumo = oa_da['Oferta y demanda agregadas, Demanda, Consumo Total'].iloc[8:].tolist()
consumo = pd.Series(consumo)

exportaciones = oa_da['Oferta y demanda agregadas, Demanda, Exportación de bienes y servicios'].iloc[8:].tolist()
exportaciones = pd.Series(exportaciones)

importaciones = oa_da['Oferta y demanda agregadas (precios corrientes), Oferta, Importación de bienes y servicios'].iloc[8:].tolist()
importaciones = pd.Series(importaciones)

exportaciones


# In[32]:


remesas.head()


# In[33]:


fechas_remesas = remesas['Título'].iloc[8:].tolist()
fechas_remesas = [fecha.date() for fecha in fechas_remesas]
fechas_remesas = pd.Series(fechas_remesas)
fechas_remesas


# In[34]:


remesas.columns


# In[35]:


dato_remesas = remesas['Remesas Familiares, Total.1'].iloc[8:].tolist()
dato_remesas= pd.Series(dato_remesas)
dato_remesas


# In[36]:


remesas = pd.concat([fechas_remesas, dato_remesas], axis = 1)
remesas.columns = ['fecha', 'remesas (numero de operaciones)']
remesas


# In[37]:


M.head(10)


# In[38]:


fechas_M = M['Título'].iloc[8:].tolist()
fechas_M = [fecha.date() for fecha in fechas_M]
fechas_M = pd.Series(fechas_M)
fechas_M


# In[39]:


dato_M = M['Fuentes y usos de la base monetaria, Usos, Billetes y monedas en circulación'].iloc[8:].tolist()
dato_M = pd.Series(dato_M)


# In[40]:


oferta_monetaria = pd.concat([fechas_M, dato_M], axis = 1)
oferta_monetaria.columns = ['fecha', 'oferta_monetaria (miles de pesos)']
oferta_monetaria


# In[41]:


pib = pd.read_excel('PIB_desestacionalizado.xlsx')


# In[42]:


pib.head()


# In[43]:


fechas_pib = pib['Título'].iloc[8:].tolist()
fechas_pib = [fecha.date() for fecha in fechas_pib]
fechas_pib = pd.Series(fechas_pib)
fechas_pib


# In[44]:


datos_pib = pib['Producto interno bruto, Producto interno bruto, a precios de mercado'].iloc[8:].tolist()
datos_pib = pd.Series(datos_pib)
datos_pib


# In[45]:


PIB = pd.concat([fechas_pib, datos_pib], axis = 1)
PIB.columns = ['trimestre', 'pib (mdp)']
PIB


# In[46]:


fechas_nuevas = []
for i in range(len(PIB['trimestre'].tolist()) - 1):
    inicio = PIB['trimestre'].tolist()[i]
    fin = PIB['trimestre'].tolist()[i + 1]
    lista_fechas = [inicio + timedelta(days=d) for d in range((fin - inicio).days + 1)] 
    fechas_nuevas.append(lista_fechas)
    
ultimo_trimestre = [PIB['trimestre'].tolist()[-1] + timedelta(days = d) for d in range(90)]
fechas_nuevas.append(ultimo_trimestre)
fechas_nuevas


# In[47]:


datos_pib = []
for i in range(len(PIB['pib (mdp)'].tolist())):
    lista_datos = np.full(len(fechas_nuevas[i]), PIB['pib (mdp)'].tolist()[i])
    lista_datos = lista_datos.tolist()
    datos_pib.append(lista_datos)


# In[48]:


pib_dia = []
fechas_gdp = []
for k in range(len(datos_pib)):
    for j in range(len(datos_pib[k])):
        pib_dia.append(datos_pib[k][j])
        fechas_gdp.append(fechas_nuevas[k][j])

pib_dia = pd.Series(pib_dia)
fechas_gdp = pd.Series(fechas_gdp)

pib_df = pd.concat([fechas_gdp, pib_dia], axis = 1)
pib_df.columns = ['fecha', 'pib (mdp)']

pib_df


# In[49]:


data = pib_df.merge(oferta_monetaria)
data = data.merge(FX)
data = data.merge(debt)
data = data.merge(remesas)
data = data.merge(pi_df)
data = data.merge(crudo)
data = data.merge(b_fiscal)
data = data.merge(indicador_act_econ)
data = data.merge(spreads)
data


# In[50]:


oa_da = pd.read_excel('OA_DA.xlsx')
oa_da.head(25)


# In[51]:


consumo_total = oa_da['Oferta y demanda agregadas, Demanda, Consumo Total'].iloc[8:]
consumo_total = pd.Series(consumo_total)
consumo_total = [consumo_total.iloc[i] for i in range(len(consumo_total))]
consumo_total


# In[52]:


fechas_oa = oa_da['Título'].iloc[8:].tolist()
fechas_oa = [fecha.date() for fecha in fechas_oa]
fechas_oa = pd.Series(fechas_oa)
fechas_oa


# In[53]:


consumo_total = oa_da['Oferta y demanda agregadas, Demanda, Consumo Total'].iloc[8:].tolist()
consumo_total = pd.Series(consumo_total)

exportaciones = oa_da['Oferta y demanda agregadas, Demanda, Exportación de bienes y servicios'].iloc[8:].tolist()
exportaciones = pd.Series(exportaciones)

importaciones = oa_da['Oferta y demanda agregadas, Oferta, Importación de bienes y servicios'].iloc[8:].tolist()
importaciones = pd.Series(importaciones)


# In[54]:


od_agregada = pd.concat([fechas_oa, consumo_total, exportaciones, importaciones], axis = 1)
od_agregada.columns = ['fecha', 'consumo (mdp)', 'exportaciones (mdp)', 'importaciones (mdp)']
od_agregada


# In[55]:


fechas_nuevas = []
for i in range(len(od_agregada['fecha'].tolist()) - 1):
    inicio = od_agregada['fecha'].tolist()[i]
    fin = od_agregada['fecha'].tolist()[i + 1]
    lista_fechas = [inicio + timedelta(days=d) for d in range((fin - inicio).days + 1)] 
    fechas_nuevas.append(lista_fechas)
    
ultimo_trimestre = [od_agregada['fecha'].tolist()[-1] + timedelta(days = d) for d in range(90)]
fechas_nuevas.append(ultimo_trimestre)
fechas_nuevas


# In[56]:


datos_oa_consumo = []
datos_oa_exportaciones = []
datos_oa_importaciones = []

for i in range(len(od_agregada['consumo (mdp)'].tolist())):
    lista_datos_consumo = np.full(len(fechas_nuevas[i]), od_agregada['consumo (mdp)'].tolist()[i])
    lista_datos_consumo = lista_datos_consumo.tolist()
    datos_oa_consumo.append(lista_datos_consumo)
    
    lista_datos_exportaciones = np.full(len(fechas_nuevas[i]), od_agregada['exportaciones (mdp)'].tolist()[i])
    lista_datos_exportaciones = lista_datos_exportaciones.tolist()
    datos_oa_exportaciones.append(lista_datos_exportaciones)
    
    lista_datos_importaciones = np.full(len(fechas_nuevas[i]), od_agregada['importaciones (mdp)'].tolist()[i])
    lista_datos_importaciones = lista_datos_importaciones.tolist()
    datos_oa_importaciones.append(lista_datos_importaciones)


# In[57]:


fechas_oada = []
consumo_oada = []
exportaciones_oada = []
importaciones_oada = []

for k in range(len(datos_oa_consumo)):
    for j in range(len(datos_oa_consumo[k])):
        fechas_oada.append(fechas_nuevas[k][j])
        consumo_oada.append(datos_oa_consumo[k][j])
        exportaciones_oada.append(datos_oa_exportaciones[k][j])
        importaciones_oada.append(datos_oa_importaciones[k][j])

fechas_oada = pd.Series(fechas_oada)
consumo_oada = pd.Series(consumo_oada)
exportaciones_oada = pd.Series(exportaciones_oada)
importaciones_oada = pd.Series(importaciones_oada)

oada_df = pd.concat([fechas_oada, consumo_oada, exportaciones_oada, importaciones_oada], axis = 1)
oada_df.columns = ['fecha', 'consumo (mdp)', 'exportaciones (mdp)', 'importaciones(mdp)']

oada_df


# In[58]:


data = data.merge(oada_df)
data['balanza comercial (mdp)'] = data['exportaciones (mdp)'] - data['importaciones(mdp)']
data


# In[59]:


inversion.head(5)


# In[60]:


fechas_inversion = inversion['Título'].iloc[8:].tolist()
fechas_inversion = [fecha.date() for fecha in fechas_inversion]
fechas_inversion = pd.Series(fechas_inversion)
fechas_inversion


# In[61]:


indice_inversion = inversion['Indice de volumen de la inversión fija bruta, Total'].iloc[8:].tolist()
indice_inversion = pd.Series(indice_inversion)

inversion_info = pd.concat([fechas_inversion, indice_inversion], axis = 1)
inversion_info.columns = ['fecha', 'indice_inversion']
inversion_info


# In[62]:


data = data.merge(inversion_info)
data


# In[63]:


data['oferta_monetaria (mdp)'] = data['oferta_monetaria (miles de pesos)']/1000


# In[64]:


data.drop(columns=['oferta_monetaria (miles de pesos)'], inplace = True)


# In[65]:


data['exportaciones_petroleo (mdp)'] = data['exportaciones_petroleo (miles de dolares)'] * data['tipo_de_cambio'] / 1000
data.drop(columns=['exportaciones_petroleo (miles de dolares)'], inplace = True)
data


# In[66]:


data['remesas (num de operaciones)'] = data['remesas (numero de operaciones)'] * 1000
data.drop(columns=['remesas (numero de operaciones)'], inplace = True)
data


# In[67]:


logs_pib = np.log(data['pib (mdp)'])


# In[68]:


data['pib en mdp(log)'] = np.log(data['pib (mdp)'])


# In[69]:


data.head()


# In[528]:


x = data['fecha'].tolist()
y = data['pib en mdp(log)'].tolist()
plt.plot_date(x, y)
plt.gcf().set_size_inches(8, 6)
plt.tight_layout()
plt.show()


# In[529]:


data.describe().round(2)


# In[530]:


data.info()


# In[531]:


data.mean()


# In[532]:


data.aggregate([min, np.mean, np.std, np.median, max]).round(2)


# los métodos de análisis estadístico están basados en los cambios a través del tiempo 

# In[533]:


data.diff().head(5)


# In[534]:


data.fecha


# In[535]:


data_alt = data.drop(['fecha'], axis = 1)
data_alt.pct_change().round(3).head()


# In[536]:


data_alt.pct_change().mean().plot(kind = 'bar', figsize = (10, 6))


# una alternativa a los cambios porcentuales son los log returns 

# In[96]:


#rets = np.log(data_alt.loc[1:] / data_alt.shift(1)).loc[1:]
#rets.head().round(3)


# In[97]:


data.set_index(['fecha'], inplace = True)


# In[98]:


sns.boxplot(data = data['pib (mdp)'])


# In[537]:


def plot_df(df, x, y, title = '', xlabel = 'Fecha', ylabel = 'Valor', dpi = 100):
    plt.figure(figsize = (16, 5), dpi = dpi)
    plt.plot(x, y, color = 'tab:red')
    plt.gca().set(title = title, xlabel = xlabel, ylabel = ylabel)
    plt.show()


# In[538]:


plot_df(data, x = data.index, y = data['pib (mdp)'], title = 'PIB mexicano en decenas de billones de pesos')


# In[540]:


plot_df(data, x = data.index, y = data['deuda (mdp)'], title = 'Deuda del gobierno mexicano en decenas de billones de pesos')


# In[585]:


plot_df(data, x = data.index, y = data['spread'], title = 'spread BONO M 10años - CETES 90días')


# In[543]:


data.columns


# In[544]:


x = data.index.values
y1 = data['oferta_monetaria (mdp)'].values

fix, ax = plt.subplots(1, 1, figsize = (16, 5), dpi = 120)
plt.fill_between(x, y1=y1, y2=-y1, alpha = 0.5, linewidth = 2, color = 'seagreen')
plt.ylim(-2510318.33, 2510318.33)
plt.title('Oferta monetaria (mdp) Two side View', fontsize = 16)
plt.hlines(y = 0, xmin = np.min(data.index), xmax = max(data.index), linewidth = .5)
plt.show()


# In[545]:


data['year'] = [d.year for d in data.index]
data['month'] = [d.strftime('%b') for d in data.index]
years = data['year'].unique()

np.random.seed(100)
mycolors = np.random.choice(list(mpl.colors.XKCD_COLORS.keys()), len(years), replace = False)

plt.figure(figsize = (16, 12), dpi = 80)
for i,y in enumerate(years):
    if i > 0:
        plt.plot('month', 'pib (mdp)', data = data.loc[data.year == y, :], color = mycolors[i], label=y)
        plt.text(data.loc[data.year == y, :].shape[0]-.9, data.loc[data.year == y][-1:].values[0], y, fontsize = 12, color = mycolors[i])
        
plt.gca().set(xlim = (data.index.min(), data.index.max()), ylim = (data['pib (mdp)'].min(), data['pib (mdp)'].max()), ylabel = 'pib', xlabel = 'mes')

plt.yticks(fontsize = 12, alpha = .7)
plt.title('lo que sea', fontsize = 20)
plt.show


# In[546]:


data.reset_index(inplace = True)


# In[547]:


data['year'] = [d.year for d in data.fecha]
data['month'] = [d.strftime('%b') for d in data.fecha]
years = data['year'].unique()

fig, axes = plt.subplots(1, 2, figsize = (20,7), dpi = 80)
sns.boxplot(x = 'year', y = 'pib (mdp)', data = data, ax = axes[0])
sns.boxplot(x = 'month', y = 'pib (mdp)', data = data)

axes[0].set_title('Diagrama de caja por año para el PIB mexicano', fontsize = 18)
axes[1].set_title('Diagrama de caja por mes para el PIB mexicano', fontsize = 18)
plt.show


# In[548]:


data['year'] = [d.year for d in data.fecha]
data['month'] = [d.strftime('%b') for d in data.fecha]
years = data['year'].unique()

fig, axes = plt.subplots(1, 2, figsize = (20,7), dpi = 80)
sns.boxplot(x = 'year', y = 'inflacion', data = data, ax = axes[0])
sns.boxplot(x = 'month', y = 'inflacion', data = data)

axes[0].set_title('Diagrama de caja por año para la inflación en México', fontsize = 18)
axes[1].set_title('Diagrama de caja por mes para la inflación en México', fontsize = 18)
plt.show


# In[556]:


datos = data[['pib (mdp)', 'deuda (mdp)', 'inflacion', 'spread']]


# In[555]:


data.columns


# In[557]:


steps = 20
datos_train = datos[:-steps]
datos_test  = datos[-steps:]

print(f"Fechas train : {datos_train.index.min()} --- {datos_train.index.max()}  (n={len(datos_train)})")
print(f"Fechas test  : {datos_test.index.min()} --- {datos_test.index.max()}  (n={len(datos_test)})")

fig, ax = plt.subplots(figsize=(9, 4))
datos_train['pib (mdp)'].plot(ax=ax, label='train')
datos_test['pib (mdp)'].plot(ax=ax, label='test')
ax.legend();


# In[558]:


forecaster = ForecasterAutoreg(
                regressor = RandomForestRegressor(random_state=123),
                lags = 10
             )

forecaster.fit(y=datos_train['pib (mdp)'])
forecaster


# In[559]:


steps = 20
predicciones = forecaster.predict(steps=steps)
predicciones.values


# In[560]:


fig, ax = plt.subplots(figsize=(9, 4))
datos_train['pib (mdp)'].plot(ax=ax, label='train')
datos_test['pib (mdp)'].plot(ax=ax, label='test')
predicciones.plot(ax=ax, label='predicciones')
ax.legend();


# In[ ]:





# In[561]:


steps = 20
forecaster = ForecasterAutoreg(
                regressor = RandomForestRegressor(random_state=123),
                lags      = 12 # Este valor será remplazado en el grid search
             )

# Lags utilizados como predictores
lags_grid = [10, 20]

# Hiperparámetros del regresor
param_grid = {'n_estimators': [100, 500],
              'max_depth': [3, 5, 10]}

resultados_grid = grid_search_forecaster(
                        forecaster         = forecaster,
                        y                  = datos_train['pib (mdp)'],
                        param_grid         = param_grid,
                        lags_grid          = lags_grid,
                        steps              = steps,
                        refit              = True,
                        metric             = 'mean_squared_error',
                        initial_train_size = int(len(datos_train)*0.5),
                        fixed_train_size   = False,
                        return_best        = True,
                        verbose            = False
                   )


# In[562]:


resultados_grid


# In[564]:


regressor = RandomForestRegressor(max_depth=10, n_estimators=100, random_state=123)
forecaster = ForecasterAutoreg(
                regressor = regressor,
                lags      = 20
             )

forecaster.fit(y=datos_train['pib (mdp)'])


# In[565]:


predicciones = forecaster.predict(steps=steps)


# In[566]:


fig, ax = plt.subplots(figsize=(9, 4))
datos_train['pib (mdp)'].plot(ax=ax, label='train')
datos_test['pib (mdp)'].plot(ax=ax, label='test')
predicciones.plot(ax=ax, label='predicciones')
ax.legend();


# In[567]:


error_mse = mean_squared_error(
                y_true = datos_test['pib (mdp)'],
                y_pred = predicciones
            )

print(f"Error de test (mse) {error_mse}")


# In[568]:


steps = 20
n_backtesting = 20*3 

metrica, predicciones_backtest = backtesting_forecaster(
                                    forecaster         = forecaster,
                                    y                  = datos['pib (mdp)'],
                                    initial_train_size = len(datos) - n_backtesting,
                                    fixed_train_size   = False,
                                    steps              = steps,
                                    refit              = True,
                                    metric             = 'mean_squared_error',
                                    verbose            = True
                                 )

print(f"Error de backtest: {metrica}")


# In[569]:


fig, ax = plt.subplots(figsize=(9, 4))
datos.loc[predicciones_backtest.index, 'pib (mdp)'].plot(ax=ax, label='test')
predicciones_backtest.plot(ax=ax, label='predicciones')
ax.legend();


# In[571]:


importancia = forecaster.get_feature_importance()
importancia


# In[572]:


from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import LassoCV
from sklearn.linear_model import ElasticNetCV
import warnings


# In[573]:


def tidy_corr_matrix(corr_mat):
    '''
    Función para convertir una matriz de correlación de pandas en formato tidy
    '''
    corr_mat = corr_mat.stack().reset_index()
    corr_mat.columns = ['variable_1','variable_2','r']
    corr_mat = corr_mat.loc[corr_mat['variable_1'] != corr_mat['variable_2'], :]
    corr_mat['abs_r'] = np.abs(corr_mat['r'])
    corr_mat = corr_mat.sort_values('abs_r', ascending=False)
    
    return(corr_mat)

corr_matrix = data.select_dtypes(include=['float64', 'int', 'object'])               .corr(method='pearson')
display(tidy_corr_matrix(corr_matrix).head(10))


# In[574]:


fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 6))

sns.heatmap(
    corr_matrix,
    square    = True,
    ax        = ax
)

ax.tick_params(labelsize = 3)


# Modelos 

# In[ ]:





# In[ ]:





# In[181]:


# Coeficientes del modelo
# ==============================================================================
df_coeficientes = pd.DataFrame(
                        {'predictor': X_train.columns,
                         'coef': modelo.coef_.flatten()}
                  )

fig, ax = plt.subplots(figsize=(11, 3.84))
ax.stem(df_coeficientes.predictor, df_coeficientes.coef, markerfmt=' ')
plt.xticks(rotation=90, ha='right', size=5)
ax.set_xlabel('variable')
ax.set_ylabel('coeficientes')
ax.set_title('Coeficientes del modelo');


# In[183]:


predicciones = modelo.predict(X=X_test)
predicciones = predicciones.flatten()
predicciones[:10]


# In[184]:


# Error de test del modelo 
# ==============================================================================
rmse_ols = mean_squared_error(
            y_true  = y_test,
            y_pred  = predicciones,
            squared = False
           )
print("")
print(f"El error (rmse) de test es: {rmse_ols}")


# In[186]:


# Error de test del modelo 
# ==============================================================================
rmse_ols = mean_squared_error(
            y_true  = y_test,
            y_pred  = predicciones,
            squared = False
           )
print("")
print(f"El error (rmse) de test es: {rmse_ols}")


# In[458]:


spreads = pd.read_excel('spreadbonos.xlsx') 
fechas = [str(fecha) for fecha in spreads['fecha'].tolist()]
fechas_dt = [datetime.strptime(str(fechas[i]), '%Y-%m-%d %H:%M:%S').date() for i in range(len(fechas))]
fechas_dt = pd.Series(fechas_dt)
spreads['fechas'] = fechas_dt
spreads.drop(['fecha', 'yield_M10', 'yield_BI90'],axis=1, inplace=True)
spreads = spreads[['fechas', 'spread']]
spreads.columns = ['fecha', 'spread']
spreads


# In[438]:


dfc = spreads.merge(pibm_df, how='left', on='fechas')
dfc.dropna(inplace= True)
dfc


# In[439]:


datos = dfc
datos.columns = ['fecha', 'spread', 'pib(mdp)']

datos['fecha'] = pd.to_datetime(datos['fecha'], format='%Y/%m/%d')
datos = datos.set_index('fecha')
datos = datos.sort_index()

fig, ax = plt.subplots(figsize=(9, 4))
datos['pib(mdp)'].plot(ax=ax, label='pib(mdp)')
datos['spread'].plot(ax=ax, label='spread yields')
ax.legend();


# In[451]:


steps = 500
datos_train = datos[:-steps]
datos_test  = datos[-steps:]

print(f"Fechas train : {datos_train.index.min()} --- {datos_train.index.max()}  (n={len(datos_train)})")
print(f"Fechas test  : {datos_test.index.min()} --- {datos_test.index.max()}  (n={len(datos_test)})")


# In[452]:


forecaster = ForecasterAutoreg(
                regressor = RandomForestRegressor(random_state=123),
                lags      = 8
             )

forecaster.fit(y=datos_train['pib(mdp)'], exog=datos_train['spread'])
forecaster


# In[453]:


predicciones = forecaster.predict(steps=steps, exog=datos_test['spread'])


# In[454]:


# Gráfico
# ==============================================================================
fig, ax=plt.subplots(figsize=(9, 4))
datos_train['pib(mdp)'].plot(ax=ax, label='train')
datos_test['pib(mdp)'].plot(ax=ax, label='test')
predicciones.plot(ax=ax, label='predicciones')
ax.legend();


# In[74]:


new_data = data[['pib (mdp)', 'inflacion']]


# In[75]:


new_data


# In[76]:


new_data.to_excel('pibinflacion.xlsx', index = False)


# In[77]:


pip install xlwings


# In[78]:


new_data


# In[79]:


import os


# In[ ]:




