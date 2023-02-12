#!/usr/bin/env python
# coding: utf-8

# In[10]:


import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

train_data = pd.read_csv("desafio_manutencao_preditiva_treino.csv")

#Converter os valores
le = LabelEncoder()
train_data['udi'] = le.fit_transform(train_data['udi'])
train_data['product_id'] = le.fit_transform(train_data['product_id'])
train_data['type'] = le.fit_transform(train_data['type'])
train_data['failure_type'] = le.fit_transform(train_data['failure_type'])

# Split the data into features (X) and target (y) #Adicionar os dados das condições de funcionamento
X_train = train_data.drop(['failure_type'], axis=1)
X_train = X_train[['udi', 'product_id', 'type', 'air_temperature_k', 'process_temperature_k', 'rotational_speed_rpm', 'torque_nm', 'tool_wear_min']]
y_train = train_data['failure_type']

# Train a classifier on the training data
classifier = DecisionTreeClassifier()
classifier.fit(X_train, y_train)


# In[19]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report


train_data = pd.read_csv("desafio_manutencao_preditiva_treino.csv")

# Preprocess the data
encoder = LabelEncoder()
train_data["failure_type"] = encoder.fit_transform(train_data["failure_type"])
train_data = train_data.drop(["udi", "product_id", "type"], axis=1)

# Include the values from air_temperature_k, process_temperature_k, rotational_speed_rpm, torque_nm and tool_wear_min in the features
X_train, X_val, y_train, y_val = train_test_split(train_data, train_data["failure_type"], test_size=0.2, random_state=42)

# Train a random forest classifier on the training data
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

test_data = pd.read_csv("desafio_manutencao_preditiva_teste.csv")

# Preprocess the test data
test_data = test_data.drop(["udi", "product_id", "type"], axis=1)
test_data["failure_type"] = encoder.transform(test_data["failure_type"])

# Predict the failure types in the test data
y_pred = clf.predict(test_data)
y_pred = encoder.inverse_transform(y_pred)

# Save the predictions to a csv file
predictions = pd.DataFrame({"failure_type": y_pred})
predictions.to_csv("predicted.csv", index=False)


# Inicialmente, decidi considerar os dados fornecidos na tabela "desafio_manutencao_preditiva_treino" para definir um padrão. Usando das informações fornecidas nas instruções do desafio e dos dados do arquivo csv extraídos pelos scripts em python, é como determinei como o caminho para realizar a previsão do tipo de falha.  Para isso, recorri a explicações em páginas que tratam sobre o tema. Após, somei meu básico conhecimento em Python e comecei a planejar como manipular os dados para obter o resultado. Sem conhecimentos na área de dados e machine learning, recorri ao auxilio da inteligência artificial; 
# De acordo com o que aprendi nas buscas e com o auxílio da inteligência artificial, obtive um código que primeiro pré-processa os dados, convertendo valores não numéricos em valores numéricos e preparando os dados para em sequencia, se ter a previsão para o desafio. Naturalmente, houve a correção e adaptação de códigos para se enquadrarem nas caracteristicas do desafio, como a retirada do valor "type" de consideração, pois seu cálculo de probabilidade esta fora do conjunto de valor considerado relevante (Apresentado no documento com dados gráficos).
# Estudando o tema, compreendi que o tipo de problema a ser resolvido é o de regressão, pois está sendo utilizado um conjunto de dados de entrada já analisados, para prever um conjunto de dados de uma situação não analizada.
# No presente momento, por ser iniciante na área, não fui capaz de determinar uma medida de performance para apresentar.
# 
