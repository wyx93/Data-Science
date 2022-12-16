#!/usr/bin/env python
# coding: utf-8

# # Data Challenge from a retail company

# #### This project consists of two sub projects:Project 1 requires the building of machine learning models to classify text files. Project 2 requires one to visualize and analyze historical data to predict spend for next 5 years.  

# ### Data Preparation

# In[1]:


#load packages
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm 
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


data = pd.read_excel("C:/Users/Stella/Desktop/temp/niagara bottling/Parts Spend.xlsx")


# ### Data Exploration

# In[3]:


# Data Sample
print(data.head())

# data types of variables
print(data.dtypes)

#size of table
print(len(data))


# In[4]:


#check if the dataset contains any NA
NA = data[data.isna().any(axis=1)]
print(NA)


# # Project 1

# ## Data Cleaning

# In[5]:


# fill NAs with empty
data['Group'] = data['Group'].fillna('') 

# Extract grouped data as grouped
grouped = data.loc[data['Group'].str.contains("GROUP.")]


# ## Modeling

# In[6]:


X=grouped['Description']
Y=grouped['Group']
# Cross validation
#split data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=21)


# In[7]:


#Vectorizing
vectorizer = CountVectorizer()
matrixTrain = vectorizer.fit_transform(X_train)
matrixTest= vectorizer.transform(X_test)
matrixAll = vectorizer.transform(data['Description'])


# In[8]:


#build a NB model
model = MultinomialNB(fit_prior=False)
model.fit(X=matrixTrain, y=Y_train)

# Prediction for test set
PredictedGroup = model.predict(X=matrixTest)

# Model performance
grouped = data.loc[data['Group'].str.contains("GROUP.")]
print (confusion_matrix(Y_test,PredictedGroup))
print (classification_report(Y_test,PredictedGroup))
print (accuracy_score(Y_test,PredictedGroup))


# In[9]:


# Prediction for complete dataset
PredictedGroup = model.predict(X=matrixAll)
data["PredictedGroup"] = PredictedGroup

# performance
grouped = data.loc[data['Group'].str.contains("GROUP.")]
print (confusion_matrix(grouped['Group'],grouped["PredictedGroup"]))
print (classification_report(grouped['Group'],grouped["PredictedGroup"]))
print (accuracy_score(grouped['Group'],grouped["PredictedGroup"]))

# Export results fot Project 1
data.to_excel("C:/Users/Stella/Desktop/temp/niagara bottling/result.xlsx")


# # Project 2

# In[10]:


# Average Spend for each Line.Age
table1=data.pivot_table(['Transaction.Value'],index='Line.Age',aggfunc='mean')
print(table1)


# In[11]:


#Average Spend for each Line
data.pivot_table(['Transaction.Value'],index='Line',aggfunc='mean')


# ## Data Cleaning

# In[12]:


# Boxplots to find outliers
data.boxplot('Transaction.Value',by = 'Line')
data.boxplot('Transaction.Value', by = 'Line.Age')


# In[13]:


# Remove outliers by line
L1=data[data['Line']=='L1'][data['Transaction.Value']>=-20000][data['Transaction.Value']<=40000]
L2=data[data['Line']=='L2'][data['Transaction.Value']<40000]
L3=data[data['Line']=='L3'][data['Transaction.Value']>=-20000][data['Transaction.Value']<25000]
data=pd.concat([L1,L2,L3])
print(data)    


# In[14]:


# Remove outliers by Line.Age
Age1=data[data['Line.Age']==1][data['Transaction.Value']>=-7000]
Age2=data[data['Line.Age']==2][data['Transaction.Value']>=-10000][data['Transaction.Value']<38000]
Age3=data[data['Line.Age']==3][data['Transaction.Value']<40000]
Age4=data[data['Line.Age']==4][data['Transaction.Value']<=40000][data['Transaction.Value']>-10000]
Age5=data[data['Line.Age']==5][data['Transaction.Value']<=20000][data['Transaction.Value']>-10000]
Age6=data[data['Line.Age']==6]
Age7=data[data['Line.Age']==7][data['Transaction.Value']<=20000]
data=pd.concat([Age1,Age2,Age3,Age4,Age5,Age7])
print(data)                 # Data is cleaned!

#Export cleaned data
data.to_excel("C:/Users/Stella/Desktop/temp/niagara bottling/cleanedData.xlsx")


# In[15]:


import datetime as dt

# add Year column
data['Transaction.Year']=pd.DatetimeIndex(data['Transaction.Date']).year

# add Quarter column
data['Transaction.Quarter']=data['Transaction.Date'].dt.quarter
print(data.head())


# ## Modeling

# In[16]:


#scatter plot for Line.Age vs Spend
sns.scatterplot(x = 'Line.Age', y = 'Transaction.Value',color= 'g',data=data)


# In[17]:


#scatter plot for Line.Age vs Avg Spend
y=[379.711467,459.289253,442.945080,588.416041,720.207014,521.376627,495.278906]
x=[1,2,3,4,5,6,7]
sns.scatterplot(x, y,color= 'g')


# ### Linear Regression

# In[18]:


X=data[['Line','Line.Age','Group','Transaction.Year']]
Y=data['Transaction.Value']

#split data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state=21)

#build OLS model on training set
model= sm.OLS(Y_train,pd.get_dummies(X_train))
olsres = model.fit()
olsres.summary()


# Transaction.Year and Line are not significant with p-value>0.05. 
# Therefore they can be eliminated from the model.

# In[19]:


X=data[['Line.Age','Group']]
Y=data['Transaction.Value']

#split data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state=21)

#build OLS model on training set
model= sm.OLS(Y_train,pd.get_dummies(X_train))
olsres = model.fit()
olsres.summary()


# ### Polynomial Regression

# In[20]:


#build polynomial model on  training data
from sklearn.preprocessing import PolynomialFeatures
import statsmodels.formula.api as smf

X = data["Line.Age"]
Y = data["Transaction.Value"] 

#split data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state=42)

#convert series to np.array
X_train=np.array(X_train)
Y_train=np.array(Y_train)

#build new dataframe
df = pd.DataFrame(columns=['y', 'x'])
df['x'] = X_train
df['y'] = Y_train

# build model for degree=2
weights = np.polyfit(X_train, Y_train, deg=2)
model = np.poly1d(weights)
results = smf.ols(formula='y ~ model(x)', data=df).fit()
results.summary()


# By comparing adjusted R-squred, we can see linear regression performs better than polynomial regression.
# Therefore I finally choose linear regression to show the relationship between line age and spend. As Line.Age increases by 1 unit, spend will on average increase by about $74.7.

# ## Spend calculation

# In[21]:


# Total part Spend by quarter
data['Trans.YQ'] = data['Transaction.Date'].dt.to_period("Q")
spendByQ = data.groupby(['Trans.YQ'])['Transaction.Value'].agg('sum')
spendByQ.columns =['Trans.YQ', 'TotalSpend']
print(spendByQ)


# In[22]:


#Total part spend by year
data.pivot_table(['Transaction.Value'],index='Transaction.Year',aggfunc='sum')

