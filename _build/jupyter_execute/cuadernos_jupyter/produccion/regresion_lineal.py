#!/usr/bin/env python
# coding: utf-8

# # Regresión Lineal

# ### CONCLUSIÓN: La regresión lineal simple no pareciera ser modelo para estos datos, puede que porque tengamos pocos datos.
# 
# Intente predecir variables correlacionalas, medianamente correlacionadas y no correlacionadas, y ninguna predice bien. Mezcle varriables climaticas con las productivas, tambien use productivas con productivas....

# In[1]:


# Imports necesarios
import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
plt.rcParams['figure.figsize'] = (16, 9)
plt.style.use('ggplot')
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score


# In[2]:


#cargamos los datos de entrada
data = pd.read_csv("datos/csv/datos_produccion_clima_35_1.csv")
#veamos cuantas dimensiones y registros contiene
data.shape


# In[3]:


#son 36 registros con 18 columnas. Veamos los primeros registros
data.head()


# In[4]:


# Ahora veamos algunas estadísticas de nuestros datos
data.describe()


# In[5]:


# Visualizamos rápidamente las caraterísticas de entrada
data.hist()
plt.show()


# # Regresión Lineal Simple

# ### Predecir a partir del promedio de lluvia anual la cantidad de lana producida. Según la matriz estas variables no estan correlacionadas. Tiene un valor de 0.19.

# In[41]:


colores=['orange','blue']
tamanios=[30,60]

f1 = data['cant_lluvia'].values
f2 = data['kilos_lana'].values

# Vamos a pintar en colores los puntos por debajo y por encima de la media de cantidad de lluvia
asignar=[]
for index, row in data.iterrows():
    if(row['cant_lluvia']>181.74):
        asignar.append(colores[0])
    else:
        asignar.append(colores[1])
    
plt.scatter(f1, f2, c=asignar, s=tamanios[0])
plt.show()


# In[42]:


# Asignamos nuestra variable de entrada X para entrenamiento y las etiquetas Y.
dataX =data[["cant_lluvia"]]
X_train = np.array(dataX)
y_train = data['kilos_lana'].values

# Creamos el objeto de Regresión Linear
regr = linear_model.LinearRegression()

# Entrenamos nuestro modelo
regr.fit(X_train, y_train)

# Hacemos las predicciones que en definitiva una línea (en este caso, al ser 2D)
y_pred = regr.predict(X_train)

# Veamos los coeficienetes obtenidos, En nuestro caso, serán la Tangente
print('Coefficients: \n', regr.coef_)
# Este es el valor donde corta el eje Y (en X=0)
print('Independent term: \n', regr.intercept_)
# Error Cuadrado Medio
print("Mean squared error: %.2f" % mean_squared_error(y_train, y_pred))
# Puntaje de Varianza. El mejor puntaje es un 1.0
print('Variance score: %.2f' % r2_score(y_train, y_pred))


# De la ecuación de la recta y = mX + b nuestra pendiente “m” es el coeficiente 7.61 y el término independiente “b” es 10429,31. Tenemos un Error Cuadrático medio enorme… por lo que en realidad este modelo no será muy bueno. Esto también se ve reflejado en el puntaje de Varianza que debería ser cercano a 1.0.

# In[43]:


plt.scatter(X_train[:,0], y_train,  c=asignar, s=tamanios[0])
plt.plot(X_train[:,0], y_pred, color='red', linewidth=3)

plt.xlabel('Promedio de Lluvia')
plt.ylabel('Compartido de Lana')
plt.title('Regresión Lineal')

plt.show()


# In[44]:


# Vamos a comprobar:
# Quiero predecir cuántos "Kilos de lana" voy a obtener por un promedio de lluvia de 200,
# según nuestro modelo, hacemos:
y_p = regr.predict([[200]])
print(round(float(y_p),2))


# ### Predecir a partir de la radiacion solar la finura de la lana producida. Según la matriz estas variables no estan correlacionadas o tienen una muy baja correlacion. Tiene un valor de -0.28.

# In[37]:


colores=['orange','blue']
tamanios=[30,60]

f1 = data['rad_solar'].values
f2 = data['finura'].values

# Vamos a pintar en colores los puntos por debajo y por encima de la media de Radiacion Solar
asignar=[]
for index, row in data.iterrows():
    if(row['rad_solar']>4248.09):
        asignar.append(colores[0])
    else:
        asignar.append(colores[1])
    
plt.scatter(f1, f2, c=asignar, s=tamanios[0])
plt.show()


# In[38]:


# Asignamos nuestra variable de entrada X para entrenamiento y las etiquetas Y.
dataX =data[["rad_solar"]]
X_train = np.array(dataX)
y_train = data['finura'].values

# Creamos el objeto de Regresión Linear
regr = linear_model.LinearRegression()

# Entrenamos nuestro modelo
regr.fit(X_train, y_train)

# Hacemos las predicciones que en definitiva una línea (en este caso, al ser 2D)
y_pred = regr.predict(X_train)

# Veamos los coeficienetes obtenidos, En nuestro caso, serán la Tangente
print('Coefficients: \n', regr.coef_)
# Este es el valor donde corta el eje Y (en X=0)
print('Independent term: \n', regr.intercept_)
# Error Cuadrado Medio
print("Mean squared error: %.2f" % mean_squared_error(y_train, y_pred))
# Puntaje de Varianza. El mejor puntaje es un 1.0
print('Variance score: %.2f' % r2_score(y_train, y_pred))


# De la ecuación de la recta y = mX + b nuestra pendiente “m” es el coeficiente -0.0009 y el término independiente “b” es 23.44. Tenemos un Error Cuadrático de 0.4… por lo que este modelo deberia ser mejor que el anterior, pero maso menos igual. Esto también se ve reflejado en el puntaje de Varianza que debería ser cercano a 1.0.

# In[39]:


plt.scatter(X_train[:,0], y_train,  c=asignar, s=tamanios[0])
plt.plot(X_train[:,0], y_pred, color='red', linewidth=3)

plt.xlabel('Cantidad de Radiacion')
plt.ylabel('Finura')
plt.title('Regresión Lineal')

plt.show()


# In[40]:


# Vamos a comprobar:
# Quiero predecir cuántos "Kilos de lana" voy a obtener con una radicion solar de 4589,
# según nuestro modelo, hacemos:
y_p = regr.predict([[4600]])
print(round(float(y_p),2))


# ### Predecir a partir de la lluvia acumulada anual la cantidad de lana producida. Según la matriz estas variables no estan correlacionadas. Tiene un valor de 0.19.

# In[30]:


colores=['orange','blue']
tamanios=[30,60]

f1 = data['acumulado_anual'].values
f2 = data['kilos_lana'].values

# Vamos a pintar en colores los puntos por debajo y por encima de la media de acumulado de lluvia anual
asignar=[]
for index, row in data.iterrows():
    if(row['acumulado_anual']>181.34):
        asignar.append(colores[0])
    else:
        asignar.append(colores[1])
    
plt.scatter(f1, f2, c=asignar, s=tamanios[0])
plt.show()


# In[31]:


# Asignamos nuestra variable de entrada X para entrenamiento y las etiquetas Y.
dataX =data[["acumulado_anual"]]
X_train = np.array(dataX)
y_train = data['kilos_lana'].values

# Creamos el objeto de Regresión Linear
regr = linear_model.LinearRegression()

# Entrenamos nuestro modelo
regr.fit(X_train, y_train)

# Hacemos las predicciones que en definitiva una línea (en este caso, al ser 2D)
y_pred = regr.predict(X_train)

# Veamos los coeficienetes obtenidos, En nuestro caso, serán la Tangente
print('Coefficients: \n', regr.coef_)
# Este es el valor donde corta el eje Y (en X=0)
print('Independent term: \n', regr.intercept_)
# Error Cuadrado Medio
print("Mean squared error: %.2f" % mean_squared_error(y_train, y_pred))
# Puntaje de Varianza. El mejor puntaje es un 1.0
print('Variance score: %.2f' % r2_score(y_train, y_pred))


# De la ecuación de la recta y = mX + b nuestra pendiente “m” es el coeficiente 7.64 y el término independiente “b” es 10427,51. Tenemos un Error Cuadrático medio enorme… por lo que en realidad este modelo no será muy bueno. Esto también se ve reflejado en el puntaje de Varianza que debería ser cercano a 1.0.

# In[32]:


plt.scatter(X_train[:,0], y_train,  c=asignar, s=tamanios[0])
plt.plot(X_train[:,0], y_pred, color='red', linewidth=3)

plt.xlabel('Acumulado de lluvia Anual')
plt.ylabel('Cantidad de Kilos de Lana')
plt.title('Regresión Lineal')

plt.show()


# In[35]:


# Vamos a comprobar:
# Quiero predecir cuántos "Kilos de lana" voy a obtener con un acumulado de lluvia de 189,
# según nuestro modelo, hacemos:
y_p = regr.predict([[189]])
print(round(float(y_p),2))


# ### Predecir a partir de la lluvia acumulada en el verano la cantidad de lana producida. Según la matriz estas variables no estan correlacionadas. Tiene un valor de -0.13.

# In[45]:


colores=['orange','blue']
tamanios=[30,60]

f1 = data['acumulado_verano'].values
f2 = data['kilos_lana'].values

# Vamos a pintar en colores los puntos por debajo y por encima de la media de acumulado de lluvia anual
asignar=[]
for index, row in data.iterrows():
    if(row['acumulado_verano']>49.51):
        asignar.append(colores[0])
    else:
        asignar.append(colores[1])
    
plt.scatter(f1, f2, c=asignar, s=tamanios[0])
plt.show()


# In[46]:


# Asignamos nuestra variable de entrada X para entrenamiento y las etiquetas Y.
dataX =data[["acumulado_verano"]]
X_train = np.array(dataX)
y_train = data['kilos_lana'].values

# Creamos el objeto de Regresión Linear
regr = linear_model.LinearRegression()

# Entrenamos nuestro modelo
regr.fit(X_train, y_train)

# Hacemos las predicciones que en definitiva una línea (en este caso, al ser 2D)
y_pred = regr.predict(X_train)

# Veamos los coeficienetes obtenidos, En nuestro caso, serán la Tangente
print('Coefficients: \n', regr.coef_)
# Este es el valor donde corta el eje Y (en X=0)
print('Independent term: \n', regr.intercept_)
# Error Cuadrado Medio
print("Mean squared error: %.2f" % mean_squared_error(y_train, y_pred))
# Puntaje de Varianza. El mejor puntaje es un 1.0
print('Variance score: %.2f' % r2_score(y_train, y_pred))


# De la ecuación de la recta y = mX + b nuestra pendiente “m” es el coeficiente -12.41 y el término independiente “b” es 12428,07. Tenemos un Error Cuadrático medio enorme… por lo que en realidad este modelo no será muy bueno. Esto también se ve reflejado en el puntaje de Varianza que debería ser cercano a 1.0.

# In[47]:


plt.scatter(X_train[:,0], y_train,  c=asignar, s=tamanios[0])
plt.plot(X_train[:,0], y_pred, color='red', linewidth=3)

plt.xlabel('Acumulado de lluvia en Verano')
plt.ylabel('Cantidad de Kilos de Lana')
plt.title('Regresión Lineal')

plt.show()


# In[48]:


# Vamos a comprobar:
# Quiero predecir cuántos "Kilos de lana" voy a obtener con un acumulado de lluvia de 189,
# según nuestro modelo, hacemos:
y_p = regr.predict([[189]])
print(round(float(y_p),2))


# OBSERVACION: lo que encuentra es que cuando menos llueve en el verano mas lana hay. !!!!!

# ### Predecir a partir de la radiacion solar la cantidad de animales que voy a esquilar. Según la matriz estas variables estan correlacionadas o tienen una mediana correlacion. Tiene un valor de 0.45.

# In[49]:


colores=['orange','blue']
tamanios=[30,60]

f1 = data['rad_solar'].values
f2 = data['esquila'].values

# Vamos a pintar en colores los puntos por debajo y por encima de la media de acumulado de lluvia anual
asignar=[]
for index, row in data.iterrows():
    if(row['rad_solar']>4298.09):
        asignar.append(colores[0])
    else:
        asignar.append(colores[1])
    
plt.scatter(f1, f2, c=asignar, s=tamanios[0])
plt.show()


# In[50]:


# Asignamos nuestra variable de entrada X para entrenamiento y las etiquetas Y.
dataX =data[["rad_solar"]]
X_train = np.array(dataX)
y_train = data['esquila'].values

# Creamos el objeto de Regresión Linear
regr = linear_model.LinearRegression()

# Entrenamos nuestro modelo
regr.fit(X_train, y_train)

# Hacemos las predicciones que en definitiva una línea (en este caso, al ser 2D)
y_pred = regr.predict(X_train)

# Veamos los coeficienetes obtenidos, En nuestro caso, serán la Tangente
print('Coefficients: \n', regr.coef_)
# Este es el valor donde corta el eje Y (en X=0)
print('Independent term: \n', regr.intercept_)
# Error Cuadrado Medio
print("Mean squared error: %.2f" % mean_squared_error(y_train, y_pred))
# Puntaje de Varianza. El mejor puntaje es un 1.0
print('Variance score: %.2f' % r2_score(y_train, y_pred))


# De la ecuación de la recta y = mX + b nuestra pendiente “m” es el coeficiente 1.52 y el término independiente “b” es -3655,53. Tenemos un Error Cuadrático medio enorme… por lo que en realidad este modelo no será muy bueno. Esto también se ve reflejado en el puntaje de Varianza que debería ser cercano a 1.0.

# In[51]:


plt.scatter(X_train[:,0], y_train,  c=asignar, s=tamanios[0])
plt.plot(X_train[:,0], y_pred, color='red', linewidth=3)

plt.xlabel('Radicion Solar')
plt.ylabel('Cantidad de Animales Esquilados')
plt.title('Regresión Lineal')

plt.show()


# In[52]:


# Vamos a comprobar:
# Quiero predecir cuántos animales esquilados voy a obtener con una radicion solar de 4566,
# según nuestro modelo, hacemos:
y_p = regr.predict([[4566]])
print(round(float(y_p),2))


# ### Predecir a partir de la cantidad de animales la cantidad de lana producida. Según la matriz estas variables estan correlacionadas positivamente. Tiene un valor de 0.64.

# In[7]:


colores=['orange','blue']
tamanios=[30,60]

f1 = data['esquila'].values
f2 = data['kilos_lana'].values

# Vamos a pintar en colores los puntos por debajo y por encima de la media de cantidad de animales
asignar=[]
for index, row in data.iterrows():
    if(row['esquila']>2910):
        asignar.append(colores[0])
    else:
        asignar.append(colores[1])
    
plt.scatter(f1, f2, c=asignar, s=tamanios[0])
plt.show()


# In[8]:


# Asignamos nuestra variable de entrada X para entrenamiento y las etiquetas Y.
dataX =data[["esquila"]]
X_train = np.array(dataX)
y_train = data['kilos_lana'].values

# Creamos el objeto de Regresión Linear
regr = linear_model.LinearRegression()

# Entrenamos nuestro modelo
regr.fit(X_train, y_train)

# Hacemos las predicciones que en definitiva una línea (en este caso, al ser 2D)
y_pred = regr.predict(X_train)

# Veamos los coeficienetes obtenidos, En nuestro caso, serán la Tangente
print('Coefficients: \n', regr.coef_)
# Este es el valor donde corta el eje Y (en X=0)
print('Independent term: \n', regr.intercept_)
# Error Cuadrado Medio
print("Mean squared error: %.2f" % mean_squared_error(y_train, y_pred))
# Puntaje de Varianza. El mejor puntaje es un 1.0
print('Variance score: %.2f' % r2_score(y_train, y_pred))


# De la ecuación de la recta y = mX + b nuestra pendiente “m” es el coeficiente 2.65 y el término independiente “b” es -4096.73. Tenemos un Error Cuadrático medio enorme… por lo que en realidad este modelo no será muy bueno. Esto también se ve reflejado en el puntaje de Varianza que debería ser cercano a 1.0.

# In[9]:


plt.scatter(X_train[:,0], y_train,  c=asignar, s=tamanios[0])
plt.plot(X_train[:,0], y_pred, color='red', linewidth=3)

plt.xlabel('Cantidad de Animales')
plt.ylabel('Cantidad de Lana')
plt.title('Regresión Lineal')

plt.show()


# In[11]:


# Vamos a comprobar:
# Quiero predecir cuántos kilos dde lana voy a obtener con una cantidad de animales de 3566,
# según nuestro modelo, hacemos:
y_p = regr.predict([[3566]])
print(round(float(y_p),2))


# ### Predecir a partir de la cantidad de ovejas preñadas la cantidad de corderos. Según la matriz estas variables estan correlacionadas positivamente. Tiene un valor de 0.58.

# In[12]:


colores=['orange','blue']
tamanios=[30,60]

f1 = data['ovejas'].values
f2 = data['corderos'].values

# Vamos a pintar en colores los puntos por debajo y por encima de la media de cantidad de animales
asignar=[]
for index, row in data.iterrows():
    if(row['ovejas']>1497):
        asignar.append(colores[0])
    else:
        asignar.append(colores[1])
    
plt.scatter(f1, f2, c=asignar, s=tamanios[0])
plt.show()


# In[13]:


# Asignamos nuestra variable de entrada X para entrenamiento y las etiquetas Y.
dataX =data[["ovejas"]]
X_train = np.array(dataX)
y_train = data['corderos'].values

# Creamos el objeto de Regresión Linear
regr = linear_model.LinearRegression()

# Entrenamos nuestro modelo
regr.fit(X_train, y_train)

# Hacemos las predicciones que en definitiva una línea (en este caso, al ser 2D)
y_pred = regr.predict(X_train)

# Veamos los coeficienetes obtenidos, En nuestro caso, serán la Tangente
print('Coefficients: \n', regr.coef_)
# Este es el valor donde corta el eje Y (en X=0)
print('Independent term: \n', regr.intercept_)
# Error Cuadrado Medio
print("Mean squared error: %.2f" % mean_squared_error(y_train, y_pred))
# Puntaje de Varianza. El mejor puntaje es un 1.0
print('Variance score: %.2f' % r2_score(y_train, y_pred))


# De la ecuación de la recta y = mX + b nuestra pendiente “m” es el coeficiente 0.37 y el término independiente “b” es -36.48. Tenemos un Error Cuadrático medio enorme… por lo que en realidad este modelo no será muy bueno. Esto también se ve reflejado en el puntaje de Varianza que debería ser cercano a 1.0.

# In[14]:


plt.scatter(X_train[:,0], y_train,  c=asignar, s=tamanios[0])
plt.plot(X_train[:,0], y_pred, color='red', linewidth=3)

plt.xlabel('Cantidad de Ovejas')
plt.ylabel('Cantidad de Corderos')
plt.title('Regresión Lineal')

plt.show()


# In[16]:


# Vamos a comprobar:
# Quiero predecir cuántos kilos dde lana voy a obtener con una cantidad de ovejas de 2566,
# según nuestro modelo, hacemos:
y_p = regr.predict([[2566]])
print(round(float(y_p),2))


# ### CONCLUSIÓN: La regresión lineal simple no pareciera ser modelo para estos datos, puede que porque tengamos pocos datos.
# 
# Intente predecir variables correlacionalas, medianamente correlacionadas y no correlacionadas, y ninguna predice bien. Mezcle varriables climaticas con las productivas, tambien use productivas con productivas....
