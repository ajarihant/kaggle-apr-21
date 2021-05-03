import pandas as pd
import numpy as np
import random
import os
import warnings
warnings.simplefilter('ignore')

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from lightgbm import LGBMClassifier

SEED = 2021
N_SPLITS = 10

def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
set_seed(SEED)

root_path = "./"
train = pd.read_csv(os.path.join(root_path, 'train.csv'))
test = pd.read_csv(os.path.join(root_path, 'test.csv'))
dataset = pd.concat([train, test], axis = 0, ignore_index = True)
TARGET = 'Survived'

dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1 
dataset['Age'] = dataset['Age'].fillna(dataset['Age'].median())
# dataset['Age']= pd.cut(dataset['Age'], [-np.inf, 20, 40, 60, 80, np.inf], right=False)
dataset['Cabin'] = dataset['Cabin'].fillna('X').map(lambda x: x[0].strip())
dataset['Ticket'] = dataset['Ticket'].fillna('X').map(lambda x:str(x).split()[0] if len(str(x).split()) > 1 else 'X')
fare_map = dataset[['Fare', 'Pclass']].dropna().groupby('Pclass').median().to_dict()
dataset['Fare'] = dataset['Fare'].fillna(dataset['Pclass'].map(fare_map['Fare']))
dataset['Fare'] = np.log1p(dataset['Fare'])
dataset['Embarked'] = dataset['Embarked'].fillna('X')
# age_map = dataset[['Age', 'Pclass']].dropna().groupby('Pclass').median().to_dict()
# dataset['Age'] = dataset['Age'].fillna(dataset['Pclass'].map(age_map['Age']))
print(dataset)

label_cols = ['Age', 'Ticket', 'Sex', 'Cabin', 'Embarked', 'Pclass', 'SibSp', 'Parch']
numerical_cols = ['Fare']#, 'FamilySize']
def label_encoder(c):
    le = LabelEncoder()
    return le.fit_transform(c)
label_encoded_df = dataset[label_cols].apply(label_encoder)
numerical_df = dataset[numerical_cols]
target_df = dataset[TARGET]
dataset = pd.concat([numerical_df, label_encoded_df, target_df], axis=1)
print(dataset)

X_train = dataset[:train.shape[0]].drop(TARGET, axis=1)
X_test = dataset[train.shape[0]:].drop(TARGET, axis=1).reset_index(drop=True)
y_train = train[TARGET]

lgbm_parameters = {
    'reg_alpha': 0.00388218567052311,
    'reg_lambda': 8.972335390951376e-05,
    'colsample_bytree': 0.18375780999902297,
    'subsample': 0.013352256062576087,
    'learning_rate': 0.002597839272059483,
    'max_depth': 44,
    'num_leaves': 15,
    'min_child_samples': 89,
    'cat_smooth': 56, 
    'cat_l2': 22.375773634793603,
    'max_bin': 33, 
    'min_data_per_group': 89
}
lgbm_parameters['metric'] = 'binary_logloss'
lgbm_parameters['objective'] = 'binary'
lgbm_parameters['n_estimators'] = 15000

dtm_oof = np.zeros(train.shape[0])
lgbm_test_pred = 0#np.zeros(len(test))
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
for fold, (train_idx, valid_idx) in enumerate(skf.split(X_train, y_train)):
    print(f"===== FOLD {fold} =====")
    X_tr, y_tr = X_train.iloc[train_idx], y_train.iloc[train_idx]
    X_va, y_va = X_train.iloc[valid_idx], y_train.iloc[valid_idx]
    lgbm_model = LGBMClassifier(**lgbm_parameters)

    lgbm_model.fit(X_tr, y_tr, eval_set = ((X_va,y_va)),verbose = 1000,categorical_feature = label_cols, early_stopping_rounds = 1000)  
    lgbm_test_pred += lgbm_model.predict_proba(X_test)[:,1]/N_SPLITS
    # lgbm_auc.append(roc_auc_score(y_valid_idx, lgbm_model.predict_proba(x_valid_idx)[:,1]))
    # lgbm_acc.append(accuracy_score(y_valid_idx,(lgbm_model.predict_proba(x_valid_idx)[:,1] > 0.5).astype(int)))


    # model.fit(X_tr, y_tr)
    dtm_oof[valid_idx] = lgbm_model.predict_proba(X_va)[:, 1]
    # dtm_preds += model.predict_proba(X_test)[: ,1] / N_SPLITS
    acc_score = accuracy_score(y_va, np.where(dtm_oof[valid_idx]>0.5, 1, 0))
    print(f"===== ACCURACY SCORE {acc_score:.6f} =====\n")
acc_score = accuracy_score(y_train, np.where(dtm_oof>0.5, 1, 0))
print(f"===== ACCURACY SCORE {acc_score:.6f} =====")

# Saving the result
submission = pd.read_csv(os.path.join(root_path, 'sample_submission.csv'))
submission['Survived'] = np.where(lgbm_test_pred > 0.5, 1, 0)
submission.to_csv("submission12-lgbm).csv", index=False)