#!/usr/bin/env python
# coding: utf-8

# # Credit Card Approval
# Bank customers apply for credit cards. But not everyone can get a credit card for a number of reasons, e.g a person may not earn enough money or may have a loan from a bank that they have to repay. Therefore, not everyone should get this credit card. The goal of this project is to create a model that allows for the acceptance and rejection of applications.

# In[48]:


import pandas as pd
import numpy as np

#Charts
import matplotlib.pyplot as plt
import seaborn as sns

#Measuring the execution time of an algorithm
import time

columns=["Gender", "Age", "Debt", "Married", "BankCustomer", "EducationLevel", "Ethnicity", "YearsEmployed", "PriorDefault", 
         "Employed", "CreditScore", "DriversLicense", "Citizen", "ZipCode", "Income", "ApprovalStatus"]

df= pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/credit-screening/crx.data', 
                names=columns, header=None)

df.head()


# # Information about data

# In[49]:


print(df.describe())

print(df.info())


# In[50]:


for column in df:
    print(column, sorted(df[column].unique()))  # Now we know that missing data is flagged "?"


# # Data preparation

# In[51]:


df=df.replace('?', np.nan)
df.fillna(df.mean(), inplace=True)
df.isna().sum()


# In[52]:


#filling in missing data

for column in df.columns:
    if df[column].dtypes == 'object':
        df.fillna(df[column].value_counts().index[0], inplace=True)
df.isna().sum()


# In[53]:


print('Number of rows with duplicates' ,df.shape[0])  #number of rows
df=df.drop_duplicates()
print('Number of rows without duplicates' ,df.shape[0])


# In[54]:


df["ApprovalStatus"].replace({"+": 1, "-": 0}, inplace=True)
print(df["ApprovalStatus"])


# In[55]:


df["ApprovalStatus"].value_counts().plot.bar()
plt.xticks([0,1], ['Decline', 'Approve'])
plt.title('Number of acceptances and rejections')


# In[56]:


from sklearn.preprocessing import LabelEncoder

coder = LabelEncoder()

df2=df.copy()


for col in df2.columns:
    if df2[col].dtypes == 'object':
        df2[col]= coder.fit_transform(df2[col])

sns.clustermap(df2.corr())


# In[57]:


df = df.drop(["DriversLicense", "ZipCode"], axis=1) #insignificant features
df


# # Modeling

# In[58]:


from sklearn.model_selection import train_test_split

X=df.drop(["ApprovalStatus"],axis=1).values
y=df["ApprovalStatus"].values


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)



# num=df.select_dtypes(include=['int64', 'float64']).columns.tolist()


# cat = df.select_dtypes(include=['object']).index.tolist()


#IMPORTANT!!! USING ColumnTransormer forces indexes, no names of columns.


df.info()

numerical=[2,7,10,12]  

categorical=[0,1,3,4,5,6,8,9,11]


# In[59]:


from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer  

    
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import ExtraTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier



 
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer

from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder    
    
    
    
    
classifiers = [
    DummyClassifier(strategy='stratified'),
    LogisticRegression(max_iter=500,solver='lbfgs'), # "solver='lbfgs" ~ in order for the algorithm to perform a specific number of iterations despite finding an optimal solution
    KNeighborsClassifier(2),
    ExtraTreeClassifier(),
    RandomForestClassifier(),
    SVC(),
    XGBClassifier(use_label_encoder=False,eval_metric='mlogloss'), # "use_label_encoder=False" ~ The use of label encoder in XGBClassifier is deprecated and will be removed in a future release."
    CatBoostClassifier(silent=True),
    LGBMClassifier(verbose=-1)
]

numeric_transformers = [StandardScaler(), MinMaxScaler(), Normalizer()]

categorical_transformers = [OrdinalEncoder(handle_unknown = 'ignore'), OneHotEncoder(handle_unknown = 'ignore')] 

#in case there is a coded value in the test set that was not in the training set ~ "handle_unknown = 'ignore'"

transformer_numerical=Pipeline(steps=[
    ('num_trans', None)
])

transformer_categorical=Pipeline(steps=[
    ('cat_trans', None)
])  

preprocessor=ColumnTransformer(transformers=[
    ('numerical', transformer_numerical, numerical),
    ('categorical', transformer_categorical, categorical)
])         
                                


# In[60]:


get_ipython().run_cell_magic('time', '', "\npipe=Pipeline([('preprocessor', preprocessor),('classifier', None)])  \n\n\nmodels_df = pd.DataFrame() #Space for scores\n\nfor model in classifiers:\n    for num_trans in numeric_transformers:\n        for cat_trans in categorical_transformers:\n            \n            pipe_params = {\n                'preprocessor__numerical__num_trans': num_trans,\n                'preprocessor__categorical__cat_trans': cat_trans,\n                'classifier': model\n            }\n            \n            \n            pipe.set_params(**pipe_params)\n            \n            \n            start_time = time.time()\n                        \n            pipe.fit(X_train, y_train)   \n            \n            end_time = time.time()\n            \n           \n            \n            \n            score = pipe.score(X_test, y_test)\n            \n            \n            \n            \n            param_dict = {\n                        'model': model.__class__.__name__,\n                        'num_trans': num_trans.__class__.__name__,\n                        'cat_trans': cat_trans.__class__.__name__,\n                        'score': score,\n                        'time_elapsed': end_time - start_time\n            }\n            \n            models_df = models_df.append(pd.DataFrame(param_dict, index=[0]))\n            \n            \nmodels_df.reset_index(drop=True, inplace=True)")


# In[61]:


models_df.sort_values('score', ascending=False)


# In[62]:


sns.boxplot(data=models_df, x='score', y='model') 
# RandomForestClassifier, 
# LogisticRegression,LGBMClassifier, XGBClassifier and CatBoostClassifier are the best


# In[63]:


sns.boxplot(data=models_df, x='time_elapsed', y='model') # CatBoostClassifier is the slowest algorithm on this data set


# # Evaluation

# In[64]:


classifiers = [
#       LogisticRegression(max_iter=500,solver='lbfgs'), # "solver='lbfgs" ~ in order for the algorithm to perform a specific number of iterations despite finding an optimal solution
      RandomForestClassifier(),
#       XGBClassifier(use_label_encoder=False,eval_metric='mlogloss'), # "use_label_encoder=False" ~ The use of label encoder in XGBClassifier is deprecated and will be removed in a future release."
#     LGBMClassifier(verbose=-1)
]

parameters={
#     'LogisticRegression':{
#         'classifier__penalty' : ['l1', 'l2'],
#         'classifier__C' : np.logspace(-4, 4, 20),
#         'classifier__solver' : ['liblinear']},
    'RandomForestClassifier':{
        'classifier__bootstrap': [True, False],
        'classifier__max_depth': [10, 20, 30, None],
        'classifier__max_features': ['auto', 'sqrt'],       
    }
#     'XGBClassifier':{
#         'classifier__learning_rate': [0.01, 0.5],
#         'classifier__max_depth': np.arange(2, 11).tolist(),
#         'classifier__min_child_weight': np.arange(0, 50).tolist()
#     }
# ,
#     'LGBMClassifier':{
#         'n_estimators': [10000],             
#         'learning_rate': [0.01, 0.3],
#         'max_depth': [3, 12]
#     }
}


numeric_transformers = [StandardScaler()]

categorical_transformers = [OneHotEncoder(handle_unknown = 'ignore')] 


# In[70]:


get_ipython().run_cell_magic('time', '', "\nfrom sklearn.model_selection import GridSearchCV\n\nimport scikitplot as skplt #roc curve\n\nfrom sklearn.metrics import classification_report\n\n\nmodels_hypers_df = pd.DataFrame() #Space for scores\n\nfor model in classifiers:\n    for num_trans in numeric_transformers:\n        for cat_trans in categorical_transformers:\n            \n            pipe_params = {\n                'preprocessor__numerical__num_trans': num_trans,\n                'preprocessor__categorical__cat_trans': cat_trans,\n                'classifier': model\n            }\n            \n            \n            pipe.set_params(**pipe_params)\n            \n            cv = GridSearchCV(pipe, parameters[model.__class__.__name__], cv=5, return_train_score=False)\n                                 \n            start_time = time.time()\n            \n            cv.fit(X_train,y_train)\n                             \n            end_time = time.time()\n            \n            \n            score=cv.score(X_test,y_test)\n            \n            #ROC curve\n            \n            y_probas = cv.predict_proba(X_test)\n    \n            skplt.metrics.plot_roc(y_test, y_probas)\n        \n            plt.title(model.__class__.__name__) \n            \n            print(cv.best_params_)\n            \n                                   \n            y_pred = cv.predict(X_test)\n            \n            print(classification_report(y_test, y_pred))\n            \n                                            \n            param_dict = {\n                        'model': model.__class__.__name__,\n                        'num_trans': num_trans.__class__.__name__,\n                        'cat_trans': cat_trans.__class__.__name__,\n                        'score': cv.best_score_,\n                        'final score': score,\n                        'time_elapsed': end_time - start_time\n            }\n            \n            models_hypers_df = models_hypers_df.append(pd.DataFrame(param_dict, index=[0]))\n            \n            \nmodels_hypers_df.reset_index(drop=True, inplace=True)\n\n\n#end-of-loops signal\n\n#time imported above\n\nimport winsound\n\n\nfor i in range(5):\n    winsound.Beep(frequency= 2500, duration= 500)\n    time.sleep(0.3)")


# In[71]:


models_hypers_df


# In[ ]:




