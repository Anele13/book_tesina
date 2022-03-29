#!/usr/bin/env python
# coding: utf-8

# # Comparar los 4 algoritmos probados

# In[1]:


import pandas as pd


# In[2]:


tabla = pd.DataFrame(columns=['algoritmo','modelo', 'descripcion', 'VE', 'MAE', 'MSE', 'RMSE','R2'],index=range(13))

tabla.iloc[0] = ('RL','prediccion1','rl que intenta predecir VP a partir de acumulado de lluvia anual',0.26,4.02,25.55,5.05,0.24)
tabla.iloc[1] = ('RL','prediccion2','rl que intenta predecir VP a partir de acumulado de lluvia en el verano',0.37,3.92,24.88,4.98,0.37)
tabla.iloc[2] = ('RL','prediccion3','rl que intenta predecir VP a partir de acumulado de lluvia anual pero agrupando por acumulado de lluvia anual',0.29,3.05,14.60,3.82,0.26)
tabla.iloc[3] = ('RL','prediccion4','rl que intenta predecir VP a partir de acumulado de lluvia en el verano pero agrupando por acumulado de lluvia anual',0.51,3.89,18.01,4.24,0.23)

tabla.iloc[4] = ('RLM','prediccion1','rl multiple que intenta predecir VP a partir de acumulado de lluvia anual y acumulado de lluvia verano',0.43,3.35,16.2,4.03,0.41)
tabla.iloc[5] = ('RLM','prediccion2','rl multiple que intenta predecir VP a partir de acumulado de lluvia anual y temperatura minima promedio',0.37,3.67,21.61,4.65,0.37)
tabla.iloc[6] = ('RLM','prediccion3','rl multiple que intenta predecir VP a partir de todas las variabes climaticas',0.44,3.40,19.16,4.38,0.44)

tabla.iloc[7] = ('KNN','prediccion1','knn que toma todas las variables climaticas y el vp',0.37,3.48,20.06,4.48,0.37)
tabla.iloc[8] = ('KNN','prediccion2','knn que toma todas las variables climaticas y el vp pero agrupadas por acumulado anual',0.30,2.65,7.14,2.67,0.14)

tabla.iloc[9] = ('arbol','prediccion1','arbol de regresion que toma todas las variables climaticas con profundidad 3',0.43,3.35,16.2,4.03,0.41)
tabla.iloc[10] = ('arbol','prediccion2','arbol de regresion que toma todas las variables climaticas con profundidad 4',0.37,3.67,21.61,4.65,0.37)
tabla.iloc[11] = ('arbol','prediccion3','arbol de regresion que toma todas las 3 variables climaticas mas correlacionadas con el VP con profundidad 4',0.37,3.67,21.61,4.65,0.37)
tabla.iloc[12] = ('arbol','prediccion4','arbol de regresion que toma todas las variables climaticas pero agrupadas por acumulado de lluvia anual con profundidad 4',0.67,3.11,10.74,3.28,0.65)

tabla


# In[3]:


tabla_sin = tabla.drop(['algoritmo','modelo','descripcion'], axis=1)
tabla_sin


# In[4]:


## Queria ver el min y max 
tabla_sin.describe()


# In[5]:


print("Mejor metrica VE es de",tabla['VE'].max())
print("Mejor metrica MAE es de",tabla['MAE'].min())
print("Mejor metrica MSE es de",tabla['MSE'].min())
print("Mejor metrica RMSE es de",tabla['RMSE'].min())
print("Mejor metrica R2 es de",tabla['R2'].max())


# ## Analisis
# 
# Si consideramos la metrica VE como la mas importante, entonces las mejores tres predicciones son:
# - La prediccion4 de arbol de regresion 0.67
# - La prediccion4 de regresion lineal simple 0.51
# - La prediccion3 de rl multiple 0.44
# 
# Si consideramos la metrica MAE como la mas importante, entonces las mejores tres predicciones son:
# - La prediccion2 de knn 2.65
# - La prediccion3 de regresion lineal simple 3.05
# - La prediccion4 de arbol de regresion 3.11
# 
# Si consideramos la metrica MSE como la mas importante, entonces las mejores tres predicciones son:
# - La prediccion2 de knn 7.14
# - La prediccion4 de arbol de regresion 10.74
# - La prediccion3 de regresion lineal simple 14.6
# 
# Si consideramos la metrica RMSE como la mas importante, entonces las mejores tres predicciones son:
# - La prediccion2 de knn 2.67
# - La prediccion4 de arbol de regresion 3.28
# - La prediccion3 de regresion lineal simple 3.82
# 
# Si consideramos la metrica R2 como la mas importante, entonces las mejores tres predicciones (2 comparten el puesto 3) son: 
# - La prediccion4 de arbol de regresion 0.65
# - La prediccion3 de rl multiple 0.44
# - La prediccion1 de rl multiple 0.41
# - La prediccion1 de arbol de regresion 0.41

# In[6]:


prediccion4_arbol = 3+1+2+2+3
prediccion4_rls= 2
prediccion3_rlm = 1+2
prediccion2_knn = 3+3+3
prediccion3_rls= 2+1+1
prediccion1_rlm = 1
prediccion1_arbol = 1


puntajes = pd.DataFrame(columns=['algoritmo','puntaje'],index=range(7))
puntajes.iloc[0]=('prediccion4_arbol',prediccion4_arbol)
puntajes.iloc[1]=('prediccion4_rls',prediccion4_rls)
puntajes.iloc[2]=('prediccion3_rlm',prediccion3_rlm)
puntajes.iloc[3]=('prediccion2_knn',prediccion2_knn)
puntajes.iloc[4]=('prediccion3_rls',prediccion3_rls)
puntajes.iloc[5]=('prediccion1_rlm',prediccion1_rlm)
puntajes.iloc[6]=('prediccion1_arbol',prediccion1_arbol)

puntajes


# ## Conclusión
# 
# Los dos algoritmos que destacan son el arbol de regresion que toma todas las variables climaticas pero agrupadas por acumulado de lluvia anual con profundidad 4; y el knn que toma todas las variables climaticas y el vp pero agrupadas por acumulado anual.
# 
# Quizas estos algoritmos mejoren mas cuando se agregen los otros 200 puntos que faltan.
