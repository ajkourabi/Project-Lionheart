#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Date: January 1 2021
#Authors: Abduljawad Kourabi, Dorian Knight
#Description: Machine learning program that evaluates inputs such as cholesterol, 
#resting bpm and sex to determine if you will experience Angina related chest pains during exercise


# # Project Lionheart 

#  This project was created for the "New Year, New Hacks" Hackathon hosted by MLH. The authors of this project 
#  wanted to create an artificial intelligence program, specifically a machine learning one, in their first hackathon. 
# 
# That was no small feat! Through the utilisation of TensorFlow, and other relvant libraries, the authors were able to create this program- which they are so excited to share! 

# In[2]:


#Import statements

import tensorflow as tf
from sklearn.model_selection import train_test_split
import pandas as pd


# In[3]:


#Data set
folder= "C:\\Users\\ajkou\\Desktop\\heart_data.csv"
angina = pd.read_csv(folder)
angina.head()


# In[4]:


#Normalizing data set
angina.columns
cols_to_norm = ["sex","cp","trestbps","chol","fbs","restecg","thalach", "oldpeak","slope","ca","thal"]
angina[cols_to_norm] = angina[cols_to_norm].apply(lambda x:(x - x.min())/(x.max()-x.min()))
angina.head()       #normalising the data- ensuring everything is weighed equally. 


# In[5]:


#Formating data set
sex = tf.feature_column.numeric_column("sex")
chest_pain = tf.feature_column.numeric_column("cp")
rest_bp = tf.feature_column.numeric_column("trestbps")
serum_cholesterol = tf.feature_column.numeric_column("chol")
fasting_blood_sugar = tf.feature_column.numeric_column("fbs")
resting_electrocardiogram = tf.feature_column.numeric_column("restecg")
max_hr = tf.feature_column.numeric_column("thalach")
induced_depression = tf.feature_column.numeric_column("oldpeak")
slope_peak = tf.feature_column.numeric_column("slope")
flourosopy_vessels = tf.feature_column.numeric_column("ca")
defect = tf.feature_column.numeric_column("thal")


# In[6]:


#The independent variables (features)
X = angina[['sex','cp','trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'oldpeak', 'slope', 'ca', 'thal']]
#Dependent variable (label)
y = angina['exang']


# In[7]:


feat_cols = [sex,chest_pain,rest_bp,serum_cholesterol,fasting_blood_sugar,resting_electrocardiogram,max_hr,induced_depression,slope_peak,flourosopy_vessels,defect]


# In[8]:


#Generating testing and training values
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.33, random_state=101)


# In[9]:


#Constants to limit how many cases the model can view at once and how many times to iterate over the cases
batch = 5
epochs = 2000


# In[11]:


#What will be used as input to train the model
input_func = tf.compat.v1.estimator.inputs.pandas_input_fn(x=X_train, y=y_train, batch_size = batch, num_epochs = epochs, shuffle = True)


# In[12]:


#Generating the linear classifier
model = tf.compat.v1.estimator.LinearClassifier(feature_columns=feat_cols,n_classes=2)


# In[13]:


#Training the model
model.train(input_fn=input_func, steps = 1000)


# In[14]:


#Using the model to make predictions on the test cases provided in the data set
pred_input_func = tf.compat.v1.estimator.inputs.pandas_input_fn( x = X_test, batch_size = batch, num_epochs = 1, shuffle = False)
predictions = model.predict(pred_input_func)
list(predictions)


# In[15]:


#Evaluates the accuracy of the model
eval_input_func = tf.compat.v1.estimator.inputs.pandas_input_fn(x=X_test, y=y_test, batch_size = batch, num_epochs=1, shuffle = False)
results = model.evaluate(eval_input_func)
results 

