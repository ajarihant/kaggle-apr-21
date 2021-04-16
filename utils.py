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
