# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("./"))

root_dir = os.getcwd()
current_dir = os.path.join(root_dir, 'titanic')

os.chdir(current_dir)

# Any results you write to the current directory are saved as output.

dataset = pd.read_csv('train.csv')
dataset_test = pd.read_csv('test.csv')


# we will train our classifier with the following features
# Numeric Features:
# - age
# - fare
# - sibSp, sibilings and spounds
# - prach, parents and children number
# Categorical Features:
# - pclass, ordinary integer (1, 2, 3)
# - sex, categories encoded as strings {female, male}
# - embarked, categories encoded as strings {S, C, Q}


# preprocessing

# we create the preprocessing pipeline for both numeric and categorical data
numeric_features = ['Age', 'SibSp', 'Parch', 'Fare']
numeric_transforms = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_features = ['Pclass', 'Sex', 'Embarked']
categorical_transforms = Pipeline(steps=[
    ('impute', SimpleImputer(strategy='constant', fill_value='missing')),
    ('scaler', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transforms, numeric_features),
        ('cat', categorical_transforms, categorical_features)
    ]
)

# Append classifier to preprocessing pipeline
# Now we have a full prediction pipeline

clf = Pipeline(steps=[
    ('preprocessor', preprocessor),
('classifier', LogisticRegression(solver='lbfgs'))
])


X = dataset.drop(labels=['PassengerId', 'Survived', 'Name', 'Ticket', 'Cabin'], axis=1)
y = dataset['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

clf.fit(X_train, y_train)

print('model score %.3f' % clf.score(X_test, y_test))

y_pred = clf.predict(dataset_test)

submission = pd.DataFrame({
    "PassengerId": dataset_test['PassengerId'],
    "Survived": y_pred
})

submission.to_csv('submission.csv', index=False)

