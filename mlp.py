#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd

# Load data
data=pd.read_csv('indian_liver_patient.csv')

print(data.head())


# In[4]:


print("the shape of the dataset is:")
data.shape


# In[5]:


# Import LabelEncoder
from sklearn import preprocessing

# Creating labelEncoder
le = preprocessing.LabelEncoder()

# Converting string labels into numbers.
data['Gender']=le.fit_transform(data['Gender'])
print(data.head())


# In[9]:


X=data[['Age','Gender','Total_Bilirubin','Direct_Bilirubin','Total_Protiens','Albumin','Albumin_and_Globulin_Ratio']]
y=data['Dataset']

# Import train_test_split function
from sklearn.model_selection import train_test_split

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)  # 70% training and 30% test


# In[10]:


data.replace([np.inf, -np.inf], np.nan, inplace=True)
data.fillna(999, inplace=True)


# In[11]:


# Import MLPClassifer 
from sklearn.neural_network import MLPClassifier

# Create model object
clf = MLPClassifier(hidden_layer_sizes=(6,5),
                    random_state=5,
                    verbose=True,
                    learning_rate_init=0.01)

# Fit data onto the model
clf.fit(X_train,y_train)


# In[14]:


# Make prediction on test dataset
ypred=clf.predict(X_test)

# Import accuracy score 
from sklearn.metrics import accuracy_score

# Calcuate accuracy
k=accuracy_score(y_test,ypred)
print("The accuracy obtained through multi-layer perceptron is")
print("accuracy=",k)


# In[ ]:




