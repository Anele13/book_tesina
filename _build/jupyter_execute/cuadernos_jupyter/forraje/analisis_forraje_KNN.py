#!/usr/bin/env python
# coding: utf-8

# # Análisis Forraje k-Nearest Neighbor

# # Conclusion: Como los datos tenian cierta agrupacion se penso que quizas este algoritmo funcionaria, pero NO. 

# ## Realizamos importaciones

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = (16, 9)
plt.style.use('ggplot')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsRegressor

from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score, mean_absolute_error


# ## PRUEBA 1

# ## Leemos nuestro archivo de entrada

# Hay que tener en cuenta que son solo 200 puntos y que solo estan relacionados con dos sondas (porque en gastre hay dos sondas del INTA pero no tienen nada de datos y tampoco tienen de los años que necesitamos)

# In[2]:


dataframe = pd.read_csv('datos/csv/datos_forraje_3_sondas.csv', sep=",")
dataframe.head(10)


# ## Estadisticas de los datos

# In[3]:


dataframe.describe()


# Son 200 registros.
# 
# El valor pastoral va del 0.0 al 29.4,con una media de 8.75 y a partir del desvío estándar podemos ver que la mayoría están entre 8.75-5.93 y 8.75+5.93.
# 
# El acumulado anual va de 119.8 hasta 304.1, con una media de 206.7 y a partir del desvío estándar podemos ver que la mayoría están entre 206.7-72.85 y 206.7+72.85.

# ## Rápidas visualizaciones

# Veamos unas gráficas simples y qué información nos aportan:

# In[4]:


dataframe.hist()
plt.show()


# Vemos que la distribuciones en la mayoria de los datos no está balanceada… esto no es bueno. Convendría tener las mismas cantidades en las salidas, para no tener resultados “tendenciosos”. 

# ## Matriz de Correlación

# In[7]:


correlation_mat = dataframe.corr()
fig, ax = plt.subplots(figsize = (12, 12))
sns.heatmap(correlation_mat, annot = True, ax = ax)


# Segun la matriz el valor pastoral esta medianamente correlacionado con la lluvia: 0.49 para anual y 0.59 para el verano. Y una baja correlacion para temperatura minimia con 0.34. 

# ## Preparamos el dataset

# Creamos nuestro X e y de entrada y los sets de entrenamiento y test.

# In[8]:


X = dataframe[['acum_anual','acum_verano', 'tem_max', 'tem_min', 'tem_med', 'viento', 'humedad']].values
y = dataframe['fn_valor_pastoral'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Veamos la forma de los datos (solo graficamos lluvia anual y valor pastoral)

# In[9]:


colores=['orange','blue']
tamanios=[30,60]

f1 = dataframe['acum_anual'].values
f2 = dataframe['fn_valor_pastoral'].values

# Vamos a pintar en colores los puntos por debajo y por encima de la media de cantidad de lluvia
asignar=[]
for index, row in dataframe.iterrows():
    if(row['acum_anual']>206):
        asignar.append(colores[0])
    else:
        asignar.append(colores[1])
    
plt.scatter(f1, f2, s=tamanios[0])
plt.show()


# ## Cómo obtener el mejor valor de k
# 
# (sobre todo importante para desempatar o elegir los puntos frontera!)
# 
# El el código que viene a continuación, vemos distintos valores k y la precisión obtenida.

# In[10]:


k_range = range(1, 50)
scores = []
for k in k_range:
    knn = KNeighborsRegressor(n_neighbors = k)
    knn.fit(X_train, y_train)
    scores.append(knn.score(X_test, y_test))
plt.figure()
plt.xlabel('k')
plt.ylabel('accuracy')
plt.scatter(k_range, scores)
plt.xticks([0,5,10,15,20,25,30,35,40,45,50])


# En la gráfica vemos que con valores k>=15 es donde mayor precisión se logra, y la misma es super baja.

# ## Creamos el Modelo

# Usemos k-Nearest Neighbor con Scikit Learn
# 
# Definimos el valor de k en 15 y creamos nuestro modelo.

# In[11]:


n_neighbors = 15

knn = KNeighborsRegressor(n_neighbors)
knn.fit(X_train, y_train)


# ### METRICAS UTILIZADAS
# 
# - *Varianza explicada*: Representa el porcentaje de la varianza de la variable de salida que es explicado por el modelo. O sea esta métrica evalúa la variación o dispersión de los puntos de datos. **explained_variance_score** La mejor puntuación posible es 1.0,los valores más bajos son peores. 
# - *Error medio absoluto*: Es la media de las diferencias absolutas entre el valor objetivo y el predicho. Al no elevar al cuadrado, no penaliza los errores grandes, lo que la hace no muy sensible a valores anómalos, por lo que no es una métrica recomendable en modelos en los que se deba prestar atención a éstos. **mean_absolute_error MAE** Mientras mas cercano a cero mejor.
# - *Error cuadrático medio*: Es simplemente la media de las diferencias entre el valor objetivo y el predicho al cuadrado. Al elevar al cuadrado los errores, magnifica los errores grandes, por lo que hay que utilizarla con cuidado cuando tenemos valores anómalos en nuestro conjunto de datos. **mean_squared_error MSE** Mientras mas cercano a cero mejor. Las unidades de la puntuación de error coinciden con las unidades del valor objetivo que se predice.
# - *Raíz cuadrada de la media del error al cuadrado*: Es igual a la raíz cuadrada de la métrica anterior. La ventaja de esta métrica es que presenta el error en las mismas unidades que la variable objetivo, lo que la hace más fácil de entender. **RMSE**. Mientras mas cercano a cero mejor. Las unidades de la puntuación de error coinciden con las unidades del valor objetivo que se predice.
# - *R cuadrado*: también llamado coeficiente de determinación. Esta métrica difiere de las anteriores, ya que compara nuestro modelo con un modelo básico que siempre devuelve como predicción la media de los valores objetivo de entrenamiento. La comparación entre estos dos modelos se realiza en base a la media de los errores al cuadrado de cada modelo. Los valores que puede tomar esta métrica van desde menos infinito a 1. **r2_score R2** Cuanto más cercano a 1 sea el valor de esta métrica, mejor será nuestro modelo. El coeficiente de determinación, también llamado R cuadrado, refleja la bondad del ajuste de un modelo a la variable que pretender explicar. Es importante saber que el resultado del coeficiente de determinación oscila entre 0 y 1. Cuanto más cerca de 1 se sitúe su valor, mayor será el ajuste del modelo a la variable que estamos intentando explicar. De forma inversa, cuanto más cerca de cero, menos ajustado estará el modelo y, por tanto, menos fiable será. 

# In[13]:



print('Accuracy of K-NN classifier on training set: {:.4f}'
     .format(knn.score(X_train, y_train)))
print('Accuracy of K-NN classifier on test set: {:.4f}'
     .format(knn.score(X_test, y_test)))

Y_pred_knn = knn.predict(X_test)

print('Varianza Explicada:')
print(str(round(explained_variance_score(y_test, Y_pred_knn),4)))
print('MAE:')
print(str(round(mean_absolute_error(y_test, Y_pred_knn),4)))
print('MSE:')
print(str(round(mean_squared_error(y_test, Y_pred_knn),4)))
print('RMSE:')
print(str(round(np.sqrt(round(mean_squared_error(y_test, Y_pred_knn),4)),4)))
print('R2:')
print(str(round(r2_score(y_test, Y_pred_knn),4)))


# - La Varianza Explicada es de 0.3746, o sea 0.37. ¿Qué significa esto? Qué los datos están dispersos un 37%, y al tener un valor por debajo de 1.0, es una metrica baja.
# - El Error medio absoluto (MAE) es de 3.4837, o sea 3.48. ¿Qué significa esto? Qué en promedio se equivoca 3.48 unidades en el valor real del VP de un punto. Considerando que el valor pastoral medido en las muestras va de 0.0 a 29.40, entonces 3.48 es medianamente aceptable. Y 3.48 está bastante cerca de 0, lo cual es bueno.
# - Error cuadrático medio (MSE) es de 20.0608, o sea 20.06. ¿Qué significa esto? Qué en promedio se equivoca 20.06 unidades en el valor real del VP de un punto. Considerando que el valor pastoral medido en las muestras va de 0.0 a 29.40, entonces 20.06 no es aceptable. Y 20.06 está lejos de 0, lo cual es malo.
# - La Raíz cuadrada del Error cuadrático medio (RMSE) es de 4.4789, o sea 4.48. ¿Qué significa esto? Qué en promedio se equivoca 4.48 unidades en el valor real del VP de un punto. Considerando que el valor pastoral medido en las muestras va de 0.0 a 29.40, entonces 4.48 es medianamente aceptable. Y 4.48 está bastante cerca de 0, lo cual es bueno.
# - El coeficiente de determinación (R2) es de 0.3718, o sea 0.37. ¿Qué significa esto? Qué el ajuste, según el cálculo del R cuadrado, es de 0.37. Esto quiere decir que es un modelo cuyas estimaciones se ajustan poco a la variable real. Aunque técnicamente no sería correcto, podríamos decir algo así como que el modelo explica en un 37% a la variable real, o que la calidad del modelo es del 37% o que el rendimiento del modelo es del 37%.  
# 
# <br/>
# 
# - La Precisión o Accuracy del modelo en el entrenamiento es de 0.4298%, o sea 43%. ¿Qué significa esto? No sé, porque no compara los valores observados con los predichos como las otras metricas. 
# - La Precisión o Accuracy del modelo en el test es de 0.3718%, o sea 37%. ¿Qué significa esto? No sé, porque no compara los valores observados con los predichos como las otras metricas. 
# 
# CONCLUSIÓN: La palabra precisión no es la correcta me parece

# ### Grafiquemos los resultados

# In[14]:


plt.plot(range(len(y_test)),y_test,label="Real")
plt.plot(range(len(Y_pred_knn)),Y_pred_knn,label="Prediccion")
 #Mostrar imagen
plt.xlabel('muestras')
# Set the y axis label of the current axis.
plt.ylabel('valor pastoral')
# Set a title of the current axe
# show a legend on the plot
plt.legend()
# Display a figure.
plt.show()c


# ## PRUEBA 2

# ## Leemos nuestro archivo de entrada

# Hay que tener en cuenta que son solo 200 puntos y que solo estan relacionados con dos sondas (porque en gastre hay dos sondas del INTA pero no tienen nada de datos y tampoco tienen de los años que necesitamos)

# In[15]:


df = pd.read_csv('datos/csv/datos_forraje_3_sondas.csv', sep=",")
df


# ### Eliminamos las columnas que no necesitamos

# In[16]:


df = df.drop(['codigo_gps', 'coordenada_completa', 'latitud','longitug', 'dist_laguna_fría','distancia_telsen','dist_gastre'], axis=1)
df


# ### Agrupamos los datos por el acumulado de lluvia anual. 
# 
# La idea es saber el promedio de valor pastoral con tantos mm de lluvia anual.

# In[17]:


df = df.groupby(by='acum_anual',as_index=False).mean()
df


# ### Estadisticas de los datos

# In[18]:


df.describe()


# ## Rápidas visualizaciones

# Veamos unas gráficas simples y qué información nos aportan:

# In[19]:


df.hist()
plt.show()


# Vemos que la distribuciones en la mayoria de los datos no está balanceada… esto no es bueno. Convendría tener las mismas cantidades en las salidas, para no tener resultados “tendenciosos”. 

# ## Preparamos el dataset

# Creamos nuestro X e y de entrada y los sets de entrenamiento y test.

# In[20]:


X = df[['acum_anual','acum_verano', 'tem_max', 'tem_min', 'tem_med', 'viento', 'humedad']].values
y = df['fn_valor_pastoral'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Veamos la forma de los datos (solo graficamos lluvia anual y valor pastoral)

# In[21]:


colores=['orange','blue']
tamanios=[30,60]

f1 = df['acum_anual'].values
f2 = df['fn_valor_pastoral'].values

# Vamos a pintar en colores los puntos por debajo y por encima de la media de cantidad de lluvia
asignar=[]
for index, row in df.iterrows():
    if(row['acum_anual']>206):
        asignar.append(colores[0])
    else:
        asignar.append(colores[1])
    
plt.scatter(f1, f2, c=asignar, s=tamanios[0])
plt.show()


# ## Cómo obtener el mejor valor de k
# 
# (sobre todo importante para desempatar o elegir los puntos frontera!)
# 
# El el código que viene a continuación, vemos distintos valores k y la precisión obtenida.

# In[22]:


k_range = range(1, 7)
scores = []
for k in k_range:
    knn = KNeighborsRegressor(n_neighbors = k)
    knn.fit(X_train, y_train)
    scores.append(knn.score(X_test, y_test))
plt.figure()
plt.xlabel('k')
plt.ylabel('accuracy')
plt.scatter(k_range, scores)
plt.xticks([0,5,10,15,20,25,30,35,40,45,50])


# En la gráfica vemos que con valores k=5 es donde mayor precisión se logra, y aca es re contra re baja.

# ## Creamos el Modelo

# Usemos k-Nearest Neighbor con Scikit Learn
# 
# Definimos el valor de k en 5 y creamos nuestro modelo.

# In[23]:


n_neighbors = 5

knn = KNeighborsRegressor(n_neighbors)
knn.fit(X_train, y_train)


# ### METRICAS UTILIZADAS
# 
# - *Varianza explicada*: Representa el porcentaje de la varianza de la variable de salida que es explicado por el modelo. O sea esta métrica evalúa la variación o dispersión de los puntos de datos. **explained_variance_score** La mejor puntuación posible es 1.0,los valores más bajos son peores. 
# - *Error medio absoluto*: Es la media de las diferencias absolutas entre el valor objetivo y el predicho. Al no elevar al cuadrado, no penaliza los errores grandes, lo que la hace no muy sensible a valores anómalos, por lo que no es una métrica recomendable en modelos en los que se deba prestar atención a éstos. **mean_absolute_error MAE** Mientras mas cercano a cero mejor.
# - *Error cuadrático medio*: Es simplemente la media de las diferencias entre el valor objetivo y el predicho al cuadrado. Al elevar al cuadrado los errores, magnifica los errores grandes, por lo que hay que utilizarla con cuidado cuando tenemos valores anómalos en nuestro conjunto de datos. **mean_squared_error MSE** Mientras mas cercano a cero mejor. Las unidades de la puntuación de error coinciden con las unidades del valor objetivo que se predice.
# - *Raíz cuadrada de la media del error al cuadrado*: Es igual a la raíz cuadrada de la métrica anterior. La ventaja de esta métrica es que presenta el error en las mismas unidades que la variable objetivo, lo que la hace más fácil de entender. **RMSE**. Mientras mas cercano a cero mejor. Las unidades de la puntuación de error coinciden con las unidades del valor objetivo que se predice.
# - *R cuadrado*: también llamado coeficiente de determinación. Esta métrica difiere de las anteriores, ya que compara nuestro modelo con un modelo básico que siempre devuelve como predicción la media de los valores objetivo de entrenamiento. La comparación entre estos dos modelos se realiza en base a la media de los errores al cuadrado de cada modelo. Los valores que puede tomar esta métrica van desde menos infinito a 1. **r2_score R2** Cuanto más cercano a 1 sea el valor de esta métrica, mejor será nuestro modelo. El coeficiente de determinación, también llamado R cuadrado, refleja la bondad del ajuste de un modelo a la variable que pretender explicar. Es importante saber que el resultado del coeficiente de determinación oscila entre 0 y 1. Cuanto más cerca de 1 se sitúe su valor, mayor será el ajuste del modelo a la variable que estamos intentando explicar. De forma inversa, cuanto más cerca de cero, menos ajustado estará el modelo y, por tanto, menos fiable será. 

# In[26]:


print('Accuracy of K-NN classifier on training set: {:.4f}'
     .format(knn.score(X_train, y_train)))
print('Accuracy of K-NN classifier on test set: {:.4f}'
     .format(knn.score(X_test, y_test)))

Y_pred_knn = knn.predict(X_test)

print('Varianza Explicada:')
print(str(round(explained_variance_score(y_test, Y_pred_knn),4)))
print('MAE:')
print(str(round(mean_absolute_error(y_test, Y_pred_knn),4)))
print('MSE:')
print(str(round(mean_squared_error(y_test, Y_pred_knn),4)))
print('RMSE:')
print(str(round(np.sqrt(round(mean_squared_error(y_test, Y_pred_knn),4)),4)))
print('R2:')
print(str(round(r2_score(y_test, Y_pred_knn),4)))


# - La Varianza Explicada es de 0.3026, o sea 0.30. ¿Qué significa esto? Qué los datos están dispersos un 30%, y al tener un valor por debajo de 1.0, es una metrica baja.
# - El Error medio absoluto (MAE) es de 2.6449, o sea 2.65. ¿Qué significa esto? Qué en promedio se equivoca 2.65 unidades en el valor real del VP de un punto. Considerando que el valor pastoral medido en las muestras va de 0.0 a 29.40, entonces 2.65 es medianamente aceptable. Y 2.65 está  cerca de 0, lo cual es bueno.
# - Error cuadrático medio (MSE) es de 7.1403, o sea 7.14. ¿Qué significa esto? Qué en promedio se equivoca 7.14 unidades en el valor real del VP de un punto. Considerando que el valor pastoral medido en las muestras va de 0.0 a 29.40, entonces 7.14 es bastante aceptable. Y 7.14 está relativamente cerca de 0, lo cual es bueno.
# - La Raíz cuadrada del Error cuadrático medio (RMSE) es de 2.6721, o sea 2.67. ¿Qué significa esto? Qué en promedio se equivoca 2.67 unidades en el valor real del VP de un punto. Considerando que el valor pastoral medido en las muestras va de 0.0 a 29.40, entonces 2.67 es medianamente aceptable. Y 2.67 está bastante cerca de 0, lo cual es bueno.
# - El coeficiente de determinación (R2) es de 0.1419, o sea 0.14. ¿Qué significa esto? Qué el ajuste, según el cálculo del R cuadrado, es de 0.14. Esto quiere decir que es un modelo cuyas estimaciones se ajustan poco a la variable real. Aunque técnicamente no sería correcto, podríamos decir algo así como que el modelo explica en un 14% a la variable real, o que la calidad del modelo es del 14% o que el rendimiento del modelo es del 14%.  
# 
# <br/>
# 
# - La Precisión o Accuracy del modelo en el entrenamiento es de 0.0287%, o sea 3%. ¿Qué significa esto? No sé, porque no compara los valores observados con los predichos como las otras metricas. 
# - La Precisión o Accuracy del modelo en el test es de 0.1419%, o sea 14%. ¿Qué significa esto? No sé, porque no compara los valores observados con los predichos como las otras metricas. 
# 
# CONCLUSIÓN: La palabra precisión no es la correcta me parece

# ### Grafiquemos los resultados

# In[28]:


plt.plot(range(len(y_test)),y_test,label="Real")
plt.plot(range(len(Y_pred_knn)),Y_pred_knn,label="Prediccion")
 #Mostrar imagen
plt.xlabel('muestras')
# Set the y axis label of the current axis.
plt.ylabel('valor pastoral')
# Set a title of the current axe
# show a legend on the plot
plt.legend()
# Display a figure.
plt.show()


# ## Comparar las 2 pruebas

# In[29]:


tabla = pd.DataFrame(columns=['modelo', 'VE', 'MAE', 'MSE', 'RMSE','R2'],index=range(2))
tabla.iloc[0] = ('prediccion1',0.37,3.48,20.06,4.48,0.37)
tabla.iloc[1] = ('prediccion2',0.30,2.65,7.14,2.67,0.14)
tabla


# Si consideramos las metricas MAE Y RMSE como las mas significativas, el mejor modelo es el de la prediccion 2.
# 
# Si consideramos la VE como la importante entonces el mejor modelo es el de la prediccion 1.
# 
# Si consideramos el R2 como el mas significativo, el modelo de la prediccion 1 es el mejor.
# 
# Según diferentes lecturas el MSE y el R2 son las mas utilizadas para regresión, y si miramos las metricas de los modelos el valor de MSE en ninguno esta cerca de 0 y el valor de R2 en ninguno esta cerca de 1, lo cual indica que ninguno funciona bien.

# Links donde hablan de las metricas y porque no usar R2 para Regresion múltiple. 
# 
# https://www.iartificial.net/regresion-lineal-con-ejemplos-en-python/ 
# 
# https://www.cienciadedatos.net/documentos/py10-regresion-lineal-python.html 
# 
# https://economipedia.com/definiciones/r-cuadrado-coeficiente-determinacion.html 
