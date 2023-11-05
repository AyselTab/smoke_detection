#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
sns.set()


# In[2]:


data=pd.read_csv(r'smoke_detection_iot.csv')

data


# In[3]:


data.describe()


# In[4]:


data.isnull().sum()


# In[5]:


data['Fire Alarm'].value_counts()


# In[6]:


data.corr()['Fire Alarm']


# In[7]:


data.columns


# In[8]:


from statsmodels.stats.outliers_influence import variance_inflation_factor

variables = data[['Unnamed: 0',
                  #'UTC', 
                  'Temperature[C]',
                  #'Humidity[%]',
                  'TVOC[ppb]',
                  'eCO2[ppm]',
                  #'Raw H2', 
                  #'Raw Ethanol',
                  #'Pressure[hPa]',
                  #'PM1.0',
                  #'PM2.5',
                  'NC0.5',
                  #'NC1.0', 
                  'NC2.5', 
                  'CNT'
    
]]
vif = pd.DataFrame()

vif["VIF"] = [variance_inflation_factor(variables.values, i) for i in range(variables.shape[1])]
vif["Features"] = variables.columns

vif_sorted= vif.sort_values(by=['VIF'], ascending=False)

vif_sorted


# In[9]:


data


# In[10]:


data= data[['NC0.5','Temperature[C]','eCO2[ppm]','TVOC[ppb]','CNT','NC2.5','Fire Alarm']]

data


# In[11]:


for i in data[['NC0.5','Temperature[C]','eCO2[ppm]','TVOC[ppb]','CNT','NC2.5']]:
    sns.boxplot(data= data, x=data[i])
    plt.show()


# In[12]:


data.describe()


# In[13]:


data= data.drop(['NC2.5','eCO2[ppm]'], axis=1)

data


# In[14]:


q1= data.quantile(0.25)
q3= data.quantile(0.75)
IQR= q3-q1

Upper=q3+1.5*IQR
Lower=q1-1.5*IQR


# In[15]:


for i in data[['NC0.5','Temperature[C]','TVOC[ppb]','CNT']]:
    data[i]=np.where(data[i]>Upper[i],Upper[i],data[i])
    data[i]=np.where(data[i]<Lower[i],Lower[i],data[i])
    sns.boxplot(data= data, x=data[i])
    plt.show()


# In[16]:


inputs  = data[['NC0.5','Temperature[C]','TVOC[ppb]','CNT']]

targets = data['Fire Alarm']


# In[17]:


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(inputs, targets, test_size=0.2, random_state=42)


# In[18]:


data


# In[19]:


from sklearn import svm
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import roc_auc_score


# In[20]:


def evaluate(model, x_test, y_test):
    
    y_pred_test = model.predict(x_test)
    y_prob_test = model.predict_proba(x_test)[:,1]
    accuracy_test = accuracy_score(y_test, y_pred_test)*100
    
    roc_probTest = roc_auc_score(y_test, y_prob_test)
    
    gini_prob_test = roc_probTest*2-1
    
    y_pred_train = model.predict(x_train)
    y_prob_train = model.predict_proba(x_train)[:,1]
    accuracy_train = accuracy_score(y_train, y_pred_train)*100
    
    roc_probTrain = roc_auc_score(y_train, y_prob_train)
    
    gini_prob_train = roc_probTrain*2-1
    

    print('Accuracy for test is', accuracy_test)
    print('Accuracy for train is', accuracy_train)
    print('gini test is',gini_prob_test*100)
    print('gini train is',gini_prob_train*100)


# In[ ]:


base_model = svm.SVC(probability=True)
base_model.fit(x_train, y_train)
base_accuracy = evaluate(base_model, x_test, y_test)


# In[ ]:


from sklearn.model_selection import RandomizedSearchCV

kernel = ['poly', 'rbf', 'sigmoid']

gamma = ['scale', 'auto'] 

C = [1, 10, 100, 1e3, 1e4, 1e5, 1e6]


random_grid = {'kernel': kernel,
               'gamma': gamma,
               'C': C}
print(random_grid)


# In[ ]:


svc_random = RandomizedSearchCV(estimator = base_model, param_distributions = random_grid, n_iter = 1, cv = 3, verbose=1, n_jobs = -1)

svc_random.fit(x_train, y_train)


# In[ ]:


svc_random.best_params_


# In[ ]:


optimized_model = svm.SVC(kernel= 'rbf', gamma = 'scale', C= 10)
optimized_model.fit(x_train,y_train)
optmized_accuracy = evaluate(optimized_model, x_test, y_test)


# In[ ]:





# In[ ]:




