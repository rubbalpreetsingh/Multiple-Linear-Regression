#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer


# In[2]:


data = pd.read_csv('data.csv')


# In[3]:


data.head(10)


# In[4]:


#In above output, we can clearly see that there are five variables, in which four variables are continuous 
#and one is categorical variable.


# In[5]:


#Extracting dependent and independent Variables:
real_x = data.iloc[:,0:4].values #if u don't use .values attribute ... then it will  gives you only location 
real_y = data.iloc[:,4].values


# In[6]:


real_y


# In[7]:


real_x


# In[8]:


# As we can see in the above output, the last column contains categorical variables 
# which are not suitable to apply directly for fitting the model. 
# So we need to encode this variable.


# In[9]:


# Encoding Dummy Variables:
# As we have one categorical variable (State), which cannot be directly applied to the model, so we will encode it. 
# To encode the categorical variable into numbers, we will use the LabelEncoder class. 
# But it is not sufficient because it still has some relational order, which may create a wrong model. 
# So in order to remove this problem, we will use OneHotEncoder, which will create the dummy variables. 
# Below is code for it:


# In[10]:


le = LabelEncoder()
real_x[:,3] = le.fit_transform(real_x[:,3])
real_x


# In[11]:


cm = ColumnTransformer([("State",OneHotEncoder(),[3])],remainder="passthrough")
real_x = cm.fit_transform(real_x)
real_x


# In[12]:


#avoiding the dummy variable trap:  
real_x = real_x[:, 1:]  
# If we do not remove the first dummy variable, then it may introduce multicollinearity in the model.


# In[13]:


training_x, test_x, training_y, test_y = train_test_split(real_x,real_y,test_size=0.2,random_state=0)


# In[14]:


training_x


# In[15]:


MLR = LinearRegression()


# In[16]:


MLR.fit(training_x,training_y)


# In[17]:


# The last step for our model is checking the performance of the model. We will do it by predicting the test set result. 
# For prediction, we will create a pred_y vector. Below is the code for it:
pred_y = MLR.predict(test_x)
pred_y


# In[18]:


test_y


# In[19]:


# Compare test_y with pred_y.. not fully accurate


# In[20]:


MLR.coef_


# In[21]:


MLR.intercept_


# In[22]:


# We can also check the score for training dataset and test dataset. Below is the code for it:
print('Train Score: ', MLR.score(training_x, training_y))  
print('Test Score: ', MLR.score(test_x, test_y))  


# # The above score tells that our model is 95% accurate with the training dataset and 93% accurate with the test dataset.
