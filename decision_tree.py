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
dataset['Cabin'] = dataset['Cabin'].fillna('X').map(lambda x: x[0].strip())
dataset['Ticket'] = dataset['Ticket'].fillna('X').map(lambda x:str(x).split()[0] if len(str(x).split()) > 1 else 'X')
fare_map = dataset[['Fare', 'Pclass']].dropna().groupby('Pclass').median().to_dict()
dataset['Fare'] = dataset['Fare'].fillna(dataset['Pclass'].map(fare_map['Fare']))
dataset['Fare'] = np.log1p(dataset['Fare'])
dataset['Embarked'] = dataset['Embarked'].fillna('X')

label_cols = ['Age', 'Ticket', 'Sex', 'Cabin', 'Embarked', 'Pclass', 'SibSp', 'Parch']
numerical_cols = ['Fare']
def label_encoder(c):
    le = LabelEncoder()
    return le.fit_transform(c)
label_encoded_df = dataset[label_cols].apply(label_encoder)
numerical_df = dataset[numerical_cols]
target_df = dataset[TARGET]
dataset = pd.concat([numerical_df, label_encoded_df, target_df], axis=1)

X_train = dataset[:train.shape[0]].drop(TARGET, axis=1)
X_test = dataset[train.shape[0]:].drop(TARGET, axis=1).reset_index(drop=True)
y_train = train[TARGET]

parameters = {
    'max_depth': np.arange(2, 5, dtype=int),
    'min_samples_leaf':  np.arange(2, 5, dtype=int)
}
classifier = DecisionTreeClassifier(random_state=SEED)
model = GridSearchCV(
    estimator=classifier,
    param_grid=parameters,
    scoring='accuracy',
    cv=10,
    n_jobs=-1)
model.fit(X_train, y_train)
best_parameters = model.best_params_
print(best_parameters)

dtm_oof = np.zeros(train.shape[0])
dtm_preds = 0
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
for fold, (train_idx, valid_idx) in enumerate(skf.split(X_train, y_train)):
    print(f"===== FOLD {fold} =====")
    X_tr, y_tr = X_train.iloc[train_idx], y_train.iloc[train_idx]
    X_va, y_va = X_train.iloc[valid_idx], y_train.iloc[valid_idx]
    model = DecisionTreeClassifier(
        max_depth=best_parameters['max_depth'],
        min_samples_leaf=best_parameters['min_samples_leaf'],
        random_state=SEED
    )
    model.fit(X_tr, y_tr)
    dtm_oof[valid_idx] = model.predict_proba(X_va)[:, 1]
    dtm_preds += model.predict_proba(X_test)[: ,1] / N_SPLITS
    acc_score = accuracy_score(y_va, np.where(dtm_oof[valid_idx]>0.5, 1, 0))
    print(f"===== ACCURACY SCORE {acc_score:.6f} =====\n")
acc_score = accuracy_score(y_train, np.where(dtm_oof>0.5, 1, 0))
print(f"===== ACCURACY SCORE {acc_score:.6f} =====")

# Saving the result
submission = pd.read_csv(os.path.join(root_path, 'sample_submission.csv'))
submission['Survived'] = np.where(dtm_preds > 0.5, 1, 0)
submission.to_csv("submission9-decision_tree.csv", index=False)