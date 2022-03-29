#!/usr/bin/env python
# coding: utf-8

# # Analizando datos

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns


# In[10]:


df = pd.read_csv('datos/csv/datos_produccion_clima_35.csv', sep=",")


# In[11]:


df


# In[52]:


df['rinde_seco'] = pd.to_numeric(df['rinde_seco'], errors='coerce')
df['finura'] = pd.to_numeric(df['finura'], errors='coerce')
df['kilo_lana_p/animal'] = pd.to_numeric(df['kilo_lana_p/animal'], errors='coerce')
df['porcentaje_paricion'] = pd.to_numeric(df['porcentaje_paricion'], errors='coerce')
df['temp_media'] = pd.to_numeric(df['temp_media'], errors='coerce')
df['temp_min'] = pd.to_numeric(df['temp_min'], errors='coerce')
df['tem_max'] = pd.to_numeric(df['tem_max'], errors='coerce')
df['humedad'] = pd.to_numeric(df['humedad'], errors='coerce')
df['cant_lluvia'] = pd.to_numeric(df['cant_lluvia'], errors='coerce')
df['rad_solar'] = pd.to_numeric(df['rad_solar'], errors='coerce')
df['vel_viento'] = pd.to_numeric(df['vel_viento'], errors='coerce')


# In[4]:


print('Cantidad de Filas y columnas:',df.shape)
print('Nombre columnas:',df.columns)


# In[12]:


df.info()


# In[13]:


df.describe()


# Hay que tener en cuenta los siguientes puntos con respecto a las matrices de correlación:
# 
# 1. Cada celda de la cuadrícula representa el valor del coeficiente de correlación entre dos variables.
# 2. El valor en la posición (a, b) representa el coeficiente de correlación entre los elementos de la fila a y la columna b. Será igual al valor en la posición (b, a)
# 3. Es una matriz cuadrada – cada fila representa una variable, y todas las columnas representan las mismas variables que las filas, de ahí el número de filas = número de columnas.
# 4. Es una matriz simétrica – esto tiene sentido porque la correlación entre a,b será la misma que la de b,a.
# 5. Todos los elementos diagonales son 1. Dado que los elementos diagonales representan la correlación de cada variable consigo misma, siempre será igual a 1.
# 6. Los marcadores de los ejes denotan el rasgo que cada uno de ellos representa.
# 7. Un valor positivo grande (cercano a 1,0) indica una fuerte correlación positiva, es decir, si el valor de una de las variables aumenta, el valor de la otra variable aumenta también.
# 8. Un valor negativo grande (cercano a -1,0) indica una fuerte correlación negativa, es decir, que el valor de una de las variables disminuye al aumentar el de la otra y viceversa.
# 9. Un valor cercano a 0 (tanto positivo como negativo) indica la ausencia de cualquier correlación entre las dos variables, y por lo tanto esas variables son independientes entre sí.
# 10. Cada celda de la matriz también está representada por sombras de un color. En este caso, los tonos más oscuros del color indican valores más pequeños, mientras que los tonos más brillantes corresponden a valores más grandes (cerca de 1). Esta escala se da con la ayuda de una barra de color en el lado derecho del gráfico.

# In[14]:


corr = df.set_index('fecha').corr()
sm.graphics.plot_corr(corr, xnames=list(corr.columns))
plt.show()


# In[15]:


correlation_mat = df.corr()
fig, ax = plt.subplots(figsize = (12, 12))
sns.heatmap(correlation_mat, annot = True, ax = ax)


# In[16]:


corr_df = df.corr(method='pearson')

corr_df.style.background_gradient(cmap='coolwarm')


# ### Cargamos dataset con datos sin ceros

# In[17]:


df = pd.read_csv('datos/csv/datos_produccion_clima_35_1.csv', sep=",")
df


# In[18]:


correlation_mat = df.corr()
fig, ax = plt.subplots(figsize = (12, 12))
sns.heatmap(correlation_mat, annot = True, ax = ax)


# In[19]:


corr_df = df.corr(method='pearson')
corr_df.style.background_gradient(cmap='coolwarm')

