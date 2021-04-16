import pandas as pd
from sklearn.preprocessing import QuantileTransformer, StandardScaler, PolynomialFeatures, LabelEncoder

import os

# root_path = '/kaggle/input/tabular-playground-series-apr-2021'
root_path = "./"

train = pd.read_csv(os.path.join(root_path, 'train.csv'))
y = train['Survived']
train = train.drop(['Survived'],1)
# print('Train dataset:\n', train.head())

test = pd.read_csv(os.path.join(root_path, 'test.csv'))
# print('Test dataset:\n', test.head())

# sample_submission = pd.read_csv(os.path.join(root_path, 'sample_submission.csv'))

dataset = pd.concat([train, test], axis = 0, ignore_index = True)
train_len = len(train)

dataset = dataset.drop(['PassengerId'],1)
print('Whole Dataset:\n', dataset.head())
# print('Types of columns:\n', dataset.dtypes)
print('Description of dataset:\n', dataset.describe(include='all'))


categorical_feature_columns = dataset.select_dtypes(exclude=['float64']).columns
# print('Categorical features:', categorical_feature_columns)

label = LabelEncoder()
for column in categorical_feature_columns:
    # label.fit(dataset[column].astype(str))
    # dataset[column] = label.transform(dataset[column])
    dataset[column] = label.fit_transform(dataset[column].astype(str))

# print('Whole Dataset:\n', dataset.head())
# print('Description of dataset:\n', dataset.describe(include='all'))
# print('Types of columns:\n', dataset.dtypes)

# print('Missing values per columns in dataset')
# for col in dataset.columns:
#     temp_col = dataset[col].isnull().sum()
#     print(f'{col}: {temp_col}')

dataset = dataset.drop(['Age', 'Fare'], 1)
print('Whole Dataset:\n', dataset.head())



from sklearn.ensemble import RandomForestClassifier
params={'max_depth' : 10, 
       'n_estimators': 100, 
        'min_samples_split': 3, 
        'min_samples_leaf': 3, 
        'bootstrap': False
       }

rf=RandomForestClassifier(**params)
rf.fit(dataset[:train_len],y)

print("CLF:\n", rf)
print("Score on train:", rf.score(dataset[:train_len],y))

ypred = rf.predict(dataset[train_len:])
print("ypred:\n", ypred)

def submit(filename):
    submission = pd.DataFrame({'PassengerId': test['PassengerId'],'Survived': ypred})
    # print("Submission:\n", submission.head())
    submission.to_csv(filename, index=False)
