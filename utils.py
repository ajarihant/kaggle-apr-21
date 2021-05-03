import pandas as pd
import numpy as np
from sklearn.preprocessing import QuantileTransformer, StandardScaler, PolynomialFeatures, LabelEncoder

import os

def one_hot_encoding(dataset, cols):
    for col in cols:
        temp = pd.get_dummies(dataset[col], prefix=col, drop_first=True)
        dataset = pd.concat([dataset, temp], axis=1)
        dataset = dataset.drop([col], 1)
    return dataset
    
def submit(filename, test, ypred):
    submission = pd.DataFrame({'PassengerId': test['PassengerId'],'Survived': ypred})
    # print("Submission:\n", submission.head())
    submission.to_csv(filename, index=False)

def label_encoder_without_NAN():
    label = LabelEncoder()
    for column in categorical_feature_columns:
        temp_notnull = dataset.loc[~dataset[column].isna()]
        target = label.fit_transform(temp_notnull[column].astype(str))
        dataset.loc[temp_notnull.index, column] = target#.astype(float)
        dataset[column] = pd.to_numeric(dataset[column], errors='coerce')
        # dataset.loc[temp_notnull.index, 'transformed_'+column] = target
        # dataset = dataset.drop([column], 1)
        # dataset[column] = label.fit_transform(dataset[column].astype(str))