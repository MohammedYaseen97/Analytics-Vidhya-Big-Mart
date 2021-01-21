#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import os
from sklearn.metrics import mean_squared_error
from math import sqrt

import config
import models


# In[4]:


try:
    train = pd.read_csv(os.path.join(config.INPUT_PATH, 'train.csv'))
    test = pd.read_csv(os.path.join(config.INPUT_PATH, 'test.csv'))
except:
    print("Files not opened!")

#Changing the year to number of years operated
train['Outlet_Establishment_Year'] = train['Outlet_Establishment_Year'] - 2013
test['Outlet_Establishment_Year'] = test['Outlet_Establishment_Year'] - 2013

# combining Item_Fat_Content misspelled
train['Item_Fat_Content'].replace(['low fat','LF','reg'],['Low Fat','Low Fat','Regular'],inplace = True)
test['Item_Fat_Content'].replace(['low fat','LF','reg'],['Low Fat','Low Fat','Regular'],inplace = True)

#Filling in missing values
train['Item_Weight'] = train.groupby('Item_Identifier')['Item_Weight'].apply(lambda x: x.fillna(x.mean()))
train['Outlet_Size'] = train.groupby('Outlet_Identifier')['Outlet_Size'].apply(lambda x: x.fillna(x.min()))

test['Item_Weight'] = test.groupby('Item_Identifier')['Item_Weight'].apply(lambda x: x.fillna(x.mean()))
test['Outlet_Size'] = test.groupby('Outlet_Identifier')['Outlet_Size'].apply(lambda x: x.fillna(x.min()))

train['Item_Weight'].fillna(train['Item_Weight'].mean(), inplace=True)
train['Outlet_Size'].fillna('Medium', inplace=True)

test['Item_Weight'].fillna(test['Item_Weight'].mean(), inplace=True)
test['Outlet_Size'].fillna('Medium', inplace=True)

#Distinguishing features for categorical and numerical
cat_features = ['Item_Fat_Content', 'Item_Type', 'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type']

num_features = [f for f in train.columns if f not in cat_features]

#One-Hot Encoding
new_df_train = pd.get_dummies(train, columns=cat_features, drop_first=True)
new_df_test = pd.get_dummies(test, columns=cat_features, drop_first=True)

#Dropping first column
new_df_train = new_df_train.drop(new_df_train.columns[0], axis=1)
new_df_test = new_df_test.drop(new_df_test.columns[0], axis=1)

final_features = [feature for feature in new_df_train.columns if feature not in ['Item_Outlet_Sales', 'Item_Identifier', 'Outlet_Identifier']]

#Shuffling training set
new_df_train = new_df_train.reset_index(drop=True)

x_train = new_df_train[final_features]
y_train = new_df_train['Item_Outlet_Sales'].values

x_test = new_df_test[final_features]

model = models.MODEL
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

output_df = pd.DataFrame({
    "Item_Identifier": test['Item_Identifier'].values,
    "Outlet_Identifier" : test['Outlet_Identifier'].values,
    "Item_Outlet_Sales" : [max(0, res) for res in y_pred]
})

output_df.to_csv(os.path.join(config.OUTPUT_PATH, 'submission.csv'), index=False)


# In[ ]:




