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

def run(fold):
    try:
        df = pd.read_csv(os.path.join(config.INPUT_PATH, 'train_folds.csv'))
    except:
        print("File not opened!")
    
    # combining Item_Fat_Content misspelled
    df['Item_Fat_Content'].replace(['low fat','LF','reg'],['Low Fat','Low Fat','Regular'],inplace = True)

    #Filling in missing values
    df['Item_Weight'] = df.groupby('Item_Identifier')['Item_Weight'].apply(lambda x: x.fillna(x.mean()))
    df['Outlet_Size'] = df.groupby('Outlet_Identifier')['Outlet_Size'].apply(lambda x: x.fillna(x.min()))
    
#     for name, group in df.groupby('Item_Identifier'):
#         print(name, group['Item_Weight'].mean())
    
    df['Item_Weight'].fillna(df['Item_Weight'].mean(), inplace=True)
    df['Outlet_Size'].fillna('Unknown', inplace=True)

    #Distinguishing features for categorical and numerical
    cat_features = ['Item_Fat_Content', 'Item_Type', 'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type']

    num_features = [f for f in df.columns if f not in cat_features]

    #One-Hot Encoding
    new_df = pd.get_dummies(df, columns=cat_features, drop_first=True)

    #Dropping first column
    new_df = new_df.drop(df.columns[0], axis=1)
#     print(new_df.head())
    
    #Dividing into train and val
    train_df = new_df[new_df.kfold != fold].reset_index(drop=True)
    val_df = new_df[new_df.kfold == fold].reset_index(drop=True)
    
    final_features = [feature for feature in new_df.columns if feature not in ['kfold', 'Item_Outlet_Sales', 'Item_Identifier', 'Outlet_Identifier']]
    
    x_train = train_df[final_features]
    y_train = train_df['Item_Outlet_Sales'].values
    
    x_val = val_df[final_features]
    y_val = val_df['Item_Outlet_Sales'].values
#     print(x_val.head())

    model = models.MODEL
    model.fit(x_train, y_train)
    
    y_pred = model.predict(x_val)
    rms = sqrt(mean_squared_error(y_val, y_pred))
    print("Fold : {}, RMSE : {}".format(fold, rms))

#Linear Regression
if __name__ == "__main__":
    run(0)
    run(1)
    run(2)
    run(3)
    run(4)