import pandas as pd
import numpy as np
import random
import os
import warnings
warnings.simplefilter('ignore')

from catboost import CatBoostClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
set_seed()

root_path = "./"
train = pd.read_csv(os.path.join(root_path, 'train.csv'))
test = pd.read_csv(os.path.join(root_path, 'test.csv'))
dataset = pd.concat([train, test], axis = 0, ignore_index = True)
# print(dataset)

dataset = pd.concat([dataset, dataset['Name'].str.split(',', expand=True)], axis=1)
dataset = dataset.rename(columns={0:'LastName', 1: 'FirstName'})
dataset['LastName'] = dataset["LastName"].str.len()
dataset['FirstName'] = dataset["FirstName"].str[1:]
dataset['FirstName'] = dataset['FirstName'].str.len()
dataset['TicketCode'] = dataset['Ticket'].str.replace('[^\w\s]','')
dataset['TicketCode'] = dataset['TicketCode'].str.replace(' ','')
dataset['TicketCode'] = dataset['TicketCode'].fillna('NA')
dataset['TicketCode'] = dataset['TicketCode'].replace('(\d)', '', regex=True)
dataset['TicketNumber'] = dataset['Ticket'].str.extract('(\d+)')
dataset['TicketNumber'] = dataset['TicketNumber'].astype(float)
dataset['TicketNumber'] = dataset['TicketNumber'].fillna(0)
dataset['AgeBin']= pd.cut(dataset['Age'], [-np.inf, 20, 40, 60, 80, np.inf], right=False)
dataset['Cabin'] = dataset["Cabin"].str[0]
dataset['Cabin'] = dataset['Cabin'].fillna('NA')
dataset['Embarked'] = dataset['Embarked'].fillna('NA')
dataset['Fare'] = dataset['Fare'].fillna(np.mean(dataset['Fare']))

dataset = dataset.drop(['Name', 'Ticket', 'Age', 'PassengerId'], axis=1)
cat_features = ['Pclass', 'Sex', 'Cabin', 'Embarked', 'TicketCode', 'AgeBin']

cont_features = [col for col in dataset.columns if col not in cat_features + ['Survived']]
features = cat_features + cont_features
print(dataset.head())

label_encoder = LabelEncoder()
for col in cat_features:
    print(col)
    dataset[col] = label_encoder.fit_transform(dataset[col].astype(str))

train = dataset.iloc[:100000,:]
test = dataset.iloc[100000:,:]
test = test.drop('Survived', axis=1)
model_results = pd.DataFrame()
test_predicts = pd.DataFrame()
folds = 5

train_oof = np.zeros((100000,))
skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
for fold, (train_idx, valid_idx) in enumerate(skf.split(train[features], train['Survived'])):
    X_train = train.iloc[train_idx]
    y_train = X_train['Survived']
    X_train = X_train.drop('Survived', axis=1)
    X_valid = train.iloc[valid_idx]
    y_valid = X_valid['Survived']
    X_valid = X_valid.drop('Survived', axis=1)

    # params = {'iterations': 10000,
    #               'use_best_model':True ,
    #               'eval_metric': 'AUC', # 'Accuracy'
    #               'loss_function':'Logloss',
    #               'od_type':'Iter',
    #               'od_wait':500,
    #               'depth': 6, # [4, 10]
    #               'l2_leaf_reg': 3,
    #               # random_strength ??
    #               'bootstrap_type': 'Bayesian',
    #               'bagging_temperature': 2,
    #               'max_bin': 254,
    #               'grow_policy': 'SymmetricTree',
    #               'cat_features': cat_features,
    #               'verbose': 500,
    #               'random_seed': 314,
    #               'task_type': "GPU",
    #      }

    model = CatBoostClassifier(
        verbose=0,
        eval_metric="Accuracy",
        random_state=42,
        cat_features=cat_features,
        task_type="GPU",
        # devices='0'
    )
    model =  model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)])
    temp_oof = model.predict(X_valid)
    train_oof[valid_idx] = temp_oof
    print(f'Fold {fold} Accuracy: ', accuracy_score(y_valid, temp_oof))

    test_predict = model.predict_proba(test)[:, 1]
    test_predict = np.where(test_predict>0.5, 1, 0)
    test_predicts['Fold '+str(fold)] = test_predict

print(f'OOF Accuracy: ', accuracy_score(train['Survived'], train_oof))
# model_results['CatBoost'] = train_oof
# print(model_results)
# print(test_predicts)

test_predicts['voting'] = np.where(test_predicts.sum(axis=1) > (folds/2), 1, 0)
submission = pd.DataFrame({'PassengerId': pd.read_csv(os.path.join(root_path, 'test.csv'))['PassengerId'],'Survived': test_predicts['voting']})
# submission.to_csv('submission7-cat_boost.csv', index=False)