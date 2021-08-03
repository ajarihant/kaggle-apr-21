import pandas as pd
import numpy as np
import os
import random
import warnings
warnings.simplefilter('ignore')

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold

import lightgbm as lgb



TARGET = 'Survived'

N_ESTIMATORS = 1000
N_SPLITS = 10
SEED = 2021
EARLY_STOPPING_ROUNDS = 100
VERBOSE = 100

def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
set_seed(SEED)


root_path = "./"
train = pd.read_csv(os.path.join(root_path, 'train.csv'))
# y = train['Survived']
# train = train.drop(['Survived'],1)
test = pd.read_csv(os.path.join(root_path, 'test.csv'))

dataset = pd.concat([train, test], axis = 0, ignore_index = True)
# train_len = len(train)
# dataset = dataset.drop(['PassengerId'], 1)
# print('*********Whole Dataset*********\n', dataset.head())

dataset['Age'] = dataset['Age'].fillna(dataset['Age'].mean())
dataset['Cabin'] = dataset['Cabin'].fillna('X').map(lambda x: x[0].strip())
dataset['Ticket'] = dataset['Ticket'].fillna('X').map(lambda x:str(x).split()[0] if len(str(x).split()) > 1 else 'X')
fare_map = dataset[['Fare', 'Pclass']].dropna().groupby('Pclass').median().to_dict()
dataset['Fare'] = dataset['Fare'].fillna(dataset['Pclass'].map(fare_map['Fare']))
dataset['Fare'] = np.log1p(dataset['Fare'])
dataset['Embarked'] = dataset['Embarked'].fillna('X')
dataset['Name'] = dataset['Name'].map(lambda x: x.split(',')[0])
# print('*********Whole Dataset*********\n', dataset.head())

label_cols = ['Name', 'Ticket', 'Sex']
onehot_cols = ['Cabin', 'Embarked']
numerical_cols = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
def label_encoder(c):
    le = LabelEncoder()
    return le.fit_transform(c)
scaler = StandardScaler()
onehot_encoded_df = pd.get_dummies(dataset[onehot_cols])
label_encoded_df = dataset[label_cols].apply(label_encoder)
numerical_df = pd.DataFrame(scaler.fit_transform(dataset[numerical_cols]), columns=numerical_cols)
target_df = dataset[TARGET]
dataset = pd.concat([numerical_df, label_encoded_df, onehot_encoded_df, target_df], axis=1)
# print('*********Whole Dataset*********\n', dataset.head())

# Light GBM
params = {
    'metric': 'binary_logloss',
    'n_estimators': N_ESTIMATORS,
    'objective': 'binary',
    'random_state': SEED,
    'learning_rate': 0.01,
    'min_child_samples': 150,
    'reg_alpha': 3e-5,
    'reg_lambda': 9e-2,
    'num_leaves': 20,
    'max_depth': 16,
    'colsample_bytree': 0.8,
    'subsample': 0.8,
    'subsample_freq': 2,
    'max_bin': 240,
}
lgb_oof = np.zeros(train.shape[0])
lgb_preds = np.zeros(test.shape[0])
feature_importances = pd.DataFrame()
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

for fold, (train_idx, valid_idx) in enumerate(skf.split(dataset, dataset[TARGET])):
    print(f"===== FOLD {fold} =====")


