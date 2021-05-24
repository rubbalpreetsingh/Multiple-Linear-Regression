#!/usr/bin/env python
# coding: utf-8

# # Step: 1- Preparation of Backward Elimination:

# In[35]:


# Importing the library: Firstly, we need to import the statsmodels.formula.api library, 
# which is used for the estimation of # various statistical models such as OLS(Ordinary Least Square). 
# Below is the code for it:


# In[36]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from sklearn.compose import ColumnTransformer


# In[37]:


data = pd.read_csv('data.csv')


# In[38]:


data.head(10)


# In[39]:


#In above output, we can clearly see that there are five variables, in which four variables are continuous 
#and one is categorical variable.


# In[40]:


#Extracting dependent and independent Variables:
real_x = data.iloc[:,0:4].values #if u don't use .values attribute ... then it will  gives you only location 
real_y = data.iloc[:,4].values


# In[41]:


real_y


# In[42]:


real_x


# In[43]:


# As we can see in the above output, the last column contains categorical variables 
# which are not suitable to apply directly for fitting the model. 
# So we need to encode this variable.


# In[44]:


# Encoding Dummy Variables:
# As we have one categorical variable (State), which cannot be directly applied to the model, so we will encode it. 
# To encode the categorical variable into numbers, we will use the LabelEncoder class. 
# But it is not sufficient because it still has some relational order, which may create a wrong model. 
# So in order to remove this problem, we will use OneHotEncoder, which will create the dummy variables. 
# Below is code for it:


# In[45]:


le = LabelEncoder()
real_x[:,3] = le.fit_transform(real_x[:,3])
real_x


# In[46]:


ct = ColumnTransformer([("State",OneHotEncoder(),[3])],remainder="passthrough")
real_x = ct.fit_transform(real_x)
real_x


# In[47]:


# To aboud the dummy Trap ----> use --> dummy variable - 1 (Use 1 less)
#avoiding the dummy variable trap:  
real_x = real_x[:, 1:]  
# If we do not remove the first dummy variable, then it may introduce multicollinearity in the model.
real_x


# In[48]:


# formula
# y = b0 + b1x1 + b2x2+....... + BnXn  ---> if x0=1 .. so b0 stands

#Code for Backward Elemenation

# Adding a column in matrix of features: As we can check in our MLR equation, there is one constant term b0, 
# but this term is not present in our matrix of features, so we need to add it manually.
# We will add a column having values x0 = 1 associated with the constant term b0.
# To add this, we will use append function of Numpy library, and will assign a value of 1. Below is the code for it.


# In[49]:


real_x = np.append(arr=np.ones((50,1)).astype(int),values=real_x,axis=1)
# Here we have used axis =1, as we wanted to add a column. For adding a row, we can use axis =0.
real_x


# In[50]:


# Now 1st column is 1 (i.s. real_x[0])
# By executing the above line of code, a new column will be added into our matrix of features,
# which will have all values equal to 1.


# # Step 2:-

# In[51]:


# 1. Now, we are actually going to apply a backward elimination process. Firstly we will create a new feature vector 
# x_opt, which will only contain a set of independent features that are significantly affecting the dependent variable.

# 2. Next, as per the Backward Elimination process, we need to choose a significant level(0.5), 
# and then need to fit the model with all possible predictors. So for fitting the model, we will create a 
# reg_OLS object of new class OLS of statsmodels library. Then we will fit it by using the fit() method.

# 3. Next we need p-value to compare with SL value, so for this we will use summary() method to get the 
# summary table of all the values. Below is the code for it:


# In[52]:


x_opt = real_x[:,[0,1,2,3,4,5]]
x_opt = np.array(x_opt,dtype=float)
reg_OLS = sm.OLS(endog=real_y,exog=x_opt).fit()
reg_OLS.summary()


# In[53]:


x_opt = real_x[:,[0,1,3,4,5]]
x_opt = np.array(x_opt,dtype=float)
reg_OLS = sm.OLS(endog=real_y,exog=x_opt).fit()
reg_OLS.summary()


# In[54]:


x_opt = real_x[:,[0,3,4,5]]
x_opt = np.array(x_opt,dtype=float)
reg_OLS = sm.OLS(endog=real_y,exog=x_opt,).fit()
reg_OLS.summary()


# In[55]:


x_opt = real_x[:,[0,3,5]]
x_opt = np.array(x_opt,dtype=float)
reg_OLS = sm.OLS(endog=real_y,exog=x_opt).fit()
reg_OLS.summary()


# In[56]:


x_opt = real_x[:,[0,3]]
x_opt = np.array(x_opt,dtype=float)
reg_OLS = sm.OLS(endog=real_y,exog=x_opt).fit()
reg_OLS.summary()


# In[57]:


# As we can see in the above output Table, only two variables are left. 
# So only the R&D independent variable is a significant variable for the prediction. 
# So we can now predict efficiently using this variable.


# In[58]:


training_x,test_x,training_y,test_y = train_test_split(x_opt,real_y,test_size=0.2,random_state=0)


# In[59]:


training_x


# In[60]:


reg = LinearRegression()


# In[61]:


reg.fit(training_x,training_y)


# In[62]:


pred_y = reg.predict(test_x)


# In[63]:


pred_y


# In[64]:


test_y


# In[65]:


print('Train Score: ', reg.score(training_x, training_y))  
print('Test Score: ', reg.score(test_x, test_y))  


# In[66]:


MLR_BE=0.9449589778363044-0.946458760778722
MLR_BE


# In[67]:


MLR = 0.9501847627493607 - 0.9347068473282446
MLR


# In[68]:


# As we can see, the training score is 94% accurate, and the test score is also 94% accurate. 
# The difference between both scores is .00149. This score is very much close to the previous score, 
# i.e., 0.0154, where we have included all the variables.


# # We got this result by using one independent variable (R&D spend) only instead of four variables. Hence, now, our model is simple and accurate.
