
from set_current_dir import set_current_dir
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix



set_current_dir('house_prices')

train_dataset = pd.read_csv('train.csv')
test_dataset = pd.read_csv('test.csv')

# preprocess

# numeric features
numeric_features = ['MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea',
              'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea',
              'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd',
              'Fireplaces', 'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch',
              '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold', 'YrSold']
numeric_transforms = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_features = ['MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1',
                   'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl',
                   'Exterior1st', 'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation',
                   'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
                   'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 'Functional',
                   'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive',
                   'PoolQC', 'Fence', 'MiscFeature', 'SaleType', 'SaleCondition']
categorical_transforms = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('scaler', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transforms, numeric_features),
    ('cat', categorical_transforms, categorical_features)
])

# append classifier to preprocessor

classifier = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0))
])

# remove unuseful columns
drop_labels = ['Id']
X = train_dataset.drop(labels=drop_labels, axis=1)
y = train_dataset['SalePrice']

# # fit local
# X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
#
# classifier.fit(X_train, y_train)
#
# y_pred = classifier.predict(X_test)

# fit all train dataset
classifier.fit(X, y)

X_test = test_dataset.drop(labels=drop_labels, axis=1)

y_pred = classifier.predict(X_test)

submission = pd.DataFrame({
    'Id': test_dataset['Id'],
    'SalePrice': y_pred
})

submission.to_csv('submission.csv', index=False)