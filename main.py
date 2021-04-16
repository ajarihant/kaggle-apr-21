import pandas as pd
import numpy as np
from sklearn.preprocessing import QuantileTransformer, StandardScaler, PolynomialFeatures, LabelEncoder

import os

seed = 42
# random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
# torch.manual_seed(seed)
# torch.cuda.manual_seed(seed)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False


# root_path = '/kaggle/input/tabular-playground-series-apr-2021'
root_path = "./"

train = pd.read_csv(os.path.join(root_path, 'train.csv'))
y = train['Survived']
train = train.drop(['Survived'],1)
# print('Train dataset:\n', train.head())

test = pd.read_csv(os.path.join(root_path, 'test.csv'))
# print('Test dataset:\n', test.head())


dataset = pd.concat([train, test], axis = 0, ignore_index = True)
train_len = len(train)

dataset = dataset.drop(['PassengerId'], 1)
# print("*********Correlation Matrix*********\n", dataset.corr())
print('*********Whole Dataset*********\n', dataset.head())
# print('Types of columns:\n', dataset.dtypes)
# print('*********Description of dataset*********\n', dataset.describe(include='all'))
print('*********Dataset Information*********\n', dataset.info())

""" Handling Missing Data"""

print('*********Unqiue values per columns in dataset*********')
for col in dataset.columns:
    temp_col = len(dataset[col].unique())#dataset[col].isnull().sum()
    print(f'{col}: {temp_col}')
    # print(df[col].unique())

print('*********Missing values per columns in dataset*********')
for col in dataset.columns:
    temp_col = dataset[col].isnull().sum()
    print(f'{col}: {temp_col}')

categorical_feature_columns = dataset.select_dtypes(exclude=['float64']).columns
label = LabelEncoder()
for column in categorical_feature_columns:
    temp_notnull = dataset.loc[~dataset[column].isna()]
    target = label.fit_transform(temp_notnull[column].astype(str))
    dataset.loc[temp_notnull.index, column] = target#.astype(float)
    dataset[column] = pd.to_numeric(dataset[column], errors='coerce')
    # dataset.loc[temp_notnull.index, 'transformed_'+column] = target
    # dataset = dataset.drop([column], 1)
    # dataset[column] = label.fit_transform(dataset[column].astype(str))
# dataset.apply(pd.to_numeric, errors='ignore')
# df.apply(pd.to_numeric, errors='ignore')
# dataset['Embarked'] = pd.to_numeric(dataset['Embarked'], errors='coerce')



# print('*********Whole Dataset*********\n', dataset.head())
# print('*********Unqiue values per columns in dataset*********')
# for col in dataset.columns:
#     temp_col = len(dataset[col].unique())#dataset[col].isnull().sum()
#     # temp_col = dataset[col].unique()#dataset[col].isnull().sum()
#     print(f'{col}: {temp_col}')
#     # print(df[col].unique())
# print('*********Missing values per columns in dataset*********')
# for col in dataset.columns:
#     temp_col = dataset[col].isnull().sum()
#     print(f'{col}: {temp_col}')
print(dataset.info())

print("*********Correlation Matrix*********\n", abs(dataset.corr()))

map_age_pclass = dataset[['Age', 'Pclass']].dropna().groupby('Pclass').mean().to_dict()
dataset['Age'] = dataset['Age'].mask(dataset['Age'].isna(), dataset['Pclass'].map(map_age_pclass['Age']))

map_age_pclass = dataset[['Fare', 'Pclass']].dropna().groupby('Pclass').mean().to_dict()
dataset['Fare'] = dataset['Fare'].mask(dataset['Fare'].isna(), dataset['Pclass'].map(map_age_pclass['Fare']))

map_age_pclass = dataset[['Ticket', 'Parch']].dropna().groupby('Parch').median().to_dict()
dataset['Ticket'] = dataset['Ticket'].mask(dataset['Ticket'].isna(), dataset['Parch'].map(map_age_pclass['Ticket']))

map_age_pclass = dataset[['Embarked', 'Sex']].dropna().groupby('Sex').median().to_dict()
dataset['Embarked_Sex'] = dataset['Embarked'].mask(dataset['Embarked'].isna(), dataset['Sex'].map(map_age_pclass['Embarked']))
map_age_pclass = dataset[['Embarked', 'Pclass']].dropna().groupby('Pclass').median().to_dict()
dataset['Embarked_Pclass'] = dataset['Embarked'].mask(dataset['Embarked'].isna(), dataset['Pclass'].map(map_age_pclass['Embarked']))
dataset = dataset.drop(['Embarked'],1)

map_age_pclass = dataset[['Cabin', 'Sex']].dropna().groupby('Sex').median().to_dict()
dataset['Cabin_Sex'] = dataset['Cabin'].mask(dataset['Cabin'].isna(), dataset['Sex'].map(map_age_pclass['Cabin']))
map_age_pclass = dataset[['Cabin', 'Pclass']].dropna().groupby('Pclass').median().to_dict()
dataset['Cabin_Pclass'] = dataset['Cabin'].mask(dataset['Cabin'].isna(), dataset['Pclass'].map(map_age_pclass['Cabin']))
dataset = dataset.drop(['Cabin'],1)

# dataset["Cabin"] = dataset["Cabin"].fillna("Nan")
# # print('Cabin Whole Dataset:\n', dataset.head())

# dataset["Ticket"] = dataset["Ticket"].fillna("Nan")


# dataset["Embarked"] = dataset["Embarked"].fillna("Nan")

# print('Whole Dataset:\n', dataset.head())

# print('Missing values per columns in dataset')
# for col in dataset.columns:
#     temp_col = dataset[col].isnull().sum()
#     print(f'{col}: {temp_col}')

# print(len(dataset.Ticket.unique()))

# print(dataset['Age'].isna().sum())

# categorical_feature_columns = dataset.select_dtypes(exclude=['float64']).columns
# # print('Categorical features:', categorical_feature_columns)

# label = LabelEncoder()
# for column in categorical_feature_columns:
#     dataset[column] = label.fit_transform(dataset[column].astype(str))

print('Whole Dataset:\n', dataset.head())
# # print('Description of dataset:\n', dataset.describe(include='all'))
# # print('Types of columns:\n', dataset.dtypes)

print('Missing values per columns in dataset')
for col in dataset.columns:
    temp_col = dataset[col].isnull().sum()
    print(f'{col}: {temp_col}')

print('*********Dataset Information*********\n', dataset.info())

# dataset = dataset.drop(['Age', 'Fare'], 1)
# print('Whole Dataset:\n', dataset.head())



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

submit("submission3-corr_based_all_missing_val.csv")