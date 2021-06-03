#!/usr/bin/env python
# coding: utf-8

# #  Predict the percentage of an student based on the no. of study hours.
# 
# 
#     
#                                                                                           
#                                                                                                  

# # Name: - Saurabh Chavan

# In[50]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics


# In[51]:


df = pd.read_csv("Student_studyhours.csv")


# In[52]:


df.info()


# In[44]:


df.describe()


# In[45]:


## plotting the scatter plot to the relationship between the dependent and independent variable

df.plot(x = 'Hours', y = 'Scores', style = '*')
plt.xlabel('Hours Studied')
plt.ylabel('Percentage Scored')
plt.show()


# In[46]:


X = df.iloc[:, :-1].values
y = df.iloc[:,1].values


# In[10]:


# splitting the dataset into test and train

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# # Linear Regression

# In[11]:


regressor = LinearRegression()
regressor.fit(X_train, y_train)


# # Potting the regression line 

# In[12]:


line = regressor.coef_*X + regressor.intercept_
plt.scatter(X, y)
plt.plot(X, line)


# In[13]:


print(X_test)
y_pred = regressor.predict(X_test)


# In[14]:


df = pd.DataFrame({'Actual': y_test , 'Predicted': y_pred})
df


# # If a student studies for 9.25 hours a day what will be his scores??

# In[33]:


sol = regressor.predict([[9.25]])
print("If a student studies for 9.25 hours a day he/she will be scoring {}".format(sol[0]))


# In[35]:


absolute_error = metrics.mean_absolute_error(y_test, y_pred)
absolute_error


# In[36]:


r2_score = regressor.score(X_test, y_test)
print(r2_score*100, '%')


# In[ ]:




