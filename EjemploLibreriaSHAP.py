#!/usr/bin/env python
# coding: utf-8

# # Ejemplo Código SHAP 

# In[2]:


pip install xgboost


# In[3]:


pip install shap


# In[4]:


#Importamos 
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier #usamos XGBClassifier porque la variable de respuesta es discreta
import shap


# In[5]:


# la data está en esta dirección https://www.kaggle.com/nareshbhat/wine-quality-binary-classification


# In[6]:


wine = pd.read_csv('wine.csv')  #La data ya está limpia, no hay datos vacíos.

wine.head()


# In[7]:


X = wine.drop('quality', axis=1) #Nuestra variable de respuesta es "Quality"
y = wine['quality']

X_train, X_test, y_train, y_test = train_test_split (X, y,test_size=0.2, random_state=42)


# In[8]:


model = XGBClassifier()
model.fit(X_train, y_train) #entrenamos el modelo
score = model.score(X_test, y_test) #Revisamos nivel de precisión
print(score) #80 % 


# In[9]:


explainer = shap.TreeExplainer(model) #Usamos shap para explicar el modelo
shap_values = explainer.shap_values(X) #Valores shap


# In[14]:


#bad
shap.initjs()
shap.force_plot(explainer.expected_value, shap_values[0, :], X.iloc[0, :])


# El gráfico force_plot, muestra la fuerza de las variables en la salida del modelo para CADA registro. En este caso tomó la primera fila del data set y representa cuáles variables influyen positivamente en que un vino sea "BUENO" en este caso residual sugar en color rojo. También, muestra los que tienen una contribución negativa en azul. 

# In[11]:


#good

shap.force_plot(explainer.expected_value, shap_values[3, :], X.iloc[3, :])


# In[12]:


shap.dependence_plot('alcohol', shap_values,X)


# En el diagrama de dispersión se grafica en el eje x los valores que tomó la variable "alcohol" en el data set y el SHAP Value de esta variable en el eje y. Por ejemplo.  un nivel de alcohol menor a 9 representa SHAP Values menores a 0.
# 
# Luego, la herramienta identifica interacción entre variables; es decir, cuál es la variable con mayor relación a la variable alcohol que incide en la variable de respuesta. En este caso, automáticamente, se identificó la variable PH. En azul se representa cuando PH es bajo (-3,1) y en rojo cuando es alto (3,5)

# In[13]:


shap.summary_plot(shap_values, X)


# ¿Cómo se interpreta? Cuando alcohol está ALTO (en rojo) y la volatile acidity en BAJO (azul), entonces se van a tener los mayores SHAP Values, es decir, es más probable que la calidad del vino (variable salida) sea BUENA.
