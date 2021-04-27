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

dataset['Cabin'] = dataset["Cabin"].str[0]
dataset['Ticket'] = dataset['Ticket'].fillna('NA')
dataset['Cabin'] = dataset['Cabin'].fillna('NA')
dataset['Embarked'] = dataset['Embarked'].fillna('NA')
dataset['Fare'] = dataset['Fare'].fillna(np.mean(dataset['Fare']))
dataset['Age'] = dataset['Age'].fillna(np.mean(dataset['Age']))
dataset = dataset.drop(['PassengerId'], axis=1)

cat_features = ['Pclass', 'Sex', 'Cabin', 'Embarked', 'Name', 'Ticket']
cont_features = [col for col in dataset.columns if col not in cat_features + ['Survived']]
features = cat_features + cont_features

label_encoder = LabelEncoder()
for col in cat_features:
    dataset[col] = label_encoder.fit_transform(dataset[col])

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
# submission = pd.DataFrame({'PassengerId': pd.read_csv(os.path.join(root_path, 'test.csv'))['PassengerId'],'Survived': test_predicts['voting']})
# submission.to_csv('submission6-cat_boost.csv', index=False)