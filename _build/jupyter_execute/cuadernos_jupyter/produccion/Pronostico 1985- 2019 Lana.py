#!/usr/bin/env python
# coding: utf-8

# # Pronostico 1985- 2019 Lana

# ## Intentamos predecir la cantidad de kilos de lana que va a sacar

# Importamos las librerias necesarias

# In[1]:


import warnings
import itertools
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
plt.style.use('fivethirtyeight')


# Cargamos datos de produccion y del clima de los ulltimos 35 años

# In[13]:


data = pd.read_csv('datos_produccion_clima_35.csv')
data


# Nos quedamos con las columnas que nos interesan

# In[14]:


data_intermedia = data[['fecha','kilos_lana']]
data_intermedia


# Convertimos la columna fecha en datetime

# In[15]:


data_intermedia['fecha'] = pd.to_datetime(data_intermedia['fecha'])
data_temp = data_intermedia
data_intermedia


# Convertimos la columna fcha en el indice del df

# In[16]:


data_temp.set_index('fecha',inplace=True)
data_temp.index.name = None
data_temp


# Visualizamos los datos

# In[18]:


data_temp.plot(figsize=(15, 6))
plt.show()


# Cuando buscamos ajustar datos de series de tiempo con un modelo ARIMA estacional, nuestro primer objetivo es encontrar los valores ARIMA(p,d,q)(P,D,Q)s que optimizan una métrica de interés. 
# 
# Asi que buscamos y seleccionamos los parámetros para el modelo de serie temporal ARIMA

# In[19]:


# Define the p, d and q parameters to take any value between 0 and 2
# Defina los parámetros p, d y q para tomar cualquier valor entre 0 y 2
p = d = q = range(0, 2)

# Generate all different combinations of p, q and q triplets
# Genere todas las diferentes combinaciones de tripletes p, d y q
pdq = list(itertools.product(p, d, q))

# Generate all different combinations of seasonal p, q and q triplets
# Genere todas las combinaciones diferentes de tripletes p, d y q estacionales
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]

print('Examples of parameter combinations for Seasonal ARIMA...')
print('Ejemplos de combinaciones de parámetros para ARIMA estacional...')
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))


# In[20]:


warnings.filterwarnings("ignore") # specify to ignore warning messages
results_aic = []
for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(data_temp,
                                            order=param,
                                            seasonal_order=param_seasonal,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)

            results = mod.fit()
            results_aic.append(results.aic)

            print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
        except:
            continue


# In[21]:


min(results_aic)
# ARIMA(0, 1, 1)x(1, 1, 1, 12)12 - AIC:167.03781106166736


# Creamos el modelo y lo entrenamos

# In[71]:


mod = sm.tsa.statespace.SARIMAX(data_temp,
                                order=(0, 1, 1),
                                seasonal_order=(1, 1, 1, 12),
                                enforce_invertibility=False)

results = mod.fit()


# Miramos comportamiento del modelo

# In[72]:


print(results.summary().tables[1])


# In[25]:


results.plot_diagnostics(figsize=(15, 12))
plt.show()


# En este caso, el diagnóstico de nuestro modelo sugiere lo siguiente:
# 
# - En el gráfico superior derecho, vemos que la línea roja KDE sigue masomenos de cerca a la línea N(0,1)(donde N(0,1) es la notación estándar para una distribución normal con media 0 y desviación estándar de 1). Ésta es una  indicación de que los residuos se distribuyen bastante normalmente.
# - El gráfico qq en la parte inferior izquierda muestra que la distribución ordenada de los residuos (puntos azules) sigue masomenos la tendencia lineal de las muestras tomadas de una distribución normal estándar con N(0,1). Nuevamente, esta es una fuerte indicación de que los residuos se distribuyen normalmente.
# - Los residuos a lo largo del tiempo (gráfico superior izquierdo) y eñ gráfico de autocorrelación (es decir, correlograma) en la parte inferior derecha, no son buenos.
# 
# Esas observaciones nos llevan a concluir que nuestro modelo produce un ajuste poco satisfactorio.

# Hacemos predicciones

# In[35]:


pred = results.get_prediction(start=pd.to_datetime('2019-12-31'), dynamic=False)
pred_ci = pred.conf_int()
pred_ci


# In[36]:


algo = pred.summary_frame(alpha=0.05)
algo


# In[37]:


ax = data_temp['1985':].plot(label='observed', figsize=(20, 15))
pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7)

ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.2)

ax.set_xlabel('Fecha')
ax.set_ylabel('Kg Lana')
plt.legend()

plt.show()


# PREDICE HORRIBLEMENTE

# In[67]:


y_forecasted = pred.predicted_mean
y_truth = data_temp['2019-12-31':]
# Compute the mean square error
# Calcule el error cuadrático medio
mse = mean_squared_error(y_truth, y_forecasted)
#mse
print('El error cuadrático medio de nuestros pronósticos es {}'.format(round(mse, 2)))


# Hago otra prediccion pero dos años adelante

# In[33]:


pred_1 = results.get_prediction(start=pd.to_datetime('2016-12-31'), end=pd.to_datetime('2022-12-31') , dynamic=False)
pred_ci_1 = pred_1.conf_int()
pred_ci_1


# In[60]:


ax = data_temp['1985':].plot(label='observed', figsize=(20, 15))
pred_1.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7)

ax.fill_between(pred_ci_1.index,
                pred_ci_1.iloc[:, 0],
                pred_ci_1.iloc[:, 1], color='k', alpha=.2)

ax.set_xlabel('Fecha')
ax.set_ylabel('Kg Lana')
plt.legend()

plt.show()


# PREDICE FEO

# Esta es otra forma de predecir

# In[40]:


pred_dynamic = results.get_prediction(start=pd.to_datetime('2019-12-31'), dynamic=True, full_results=True)
pred_dynamic_ci = pred_dynamic.conf_int()
pred_dynamic_ci


# In[62]:


ax = data_temp['1985':].plot(label='observed', figsize=(20, 15))
pred_dynamic.predicted_mean.plot(label='Dynamic Forecast', ax=ax)

ax.fill_between(pred_dynamic_ci.index,
                pred_dynamic_ci.iloc[:, 0],
                pred_dynamic_ci.iloc[:, 1], color='k', alpha=.25)

ax.fill_betweenx(ax.get_ylim(), pd.to_datetime('2018-01-01'), data_temp.index[-1],
                 alpha=.1, zorder=-1)

ax.set_xlabel('Fecha')
ax.set_ylabel('Kg Lana')

plt.legend()
plt.show()


# In[70]:


# Extract the predicted and true values of our time series
# Extraiga los valores predichos y verdaderos de nuestra serie de tiempo
y_forecasted = pred_dynamic.predicted_mean
y_truth = data_temp['2019-01-01':]

# Compute the mean square error
# Calcule el error cuadrático medio
mse = mean_squared_error(y_truth, y_forecasted)
print('El error cuadrático medio de nuestros pronósticos es {}'.format(round(mse, 2)))


# Predice mal!

# Predecimos 10 años en el futuro

# In[64]:


# Get forecast 500 steps ahead in future
# Obtenga una previsión de 10 pasos adelante en el futuro
pred_uc = results.get_forecast(steps=10)

# Get confidence intervals of forecasts
# Obtenga intervalos de confianza de los pronósticos
pred_ci = pred_uc.conf_int()


# In[65]:


ax = data_temp.plot(label='observed', figsize=(20, 15))
pred_uc.predicted_mean.plot(ax=ax, label='Forecast')
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.25)

ax.set_xlabel('Año')
ax.set_ylabel('Kg Lana')

plt.legend()
plt.show()


# In[ ]:




