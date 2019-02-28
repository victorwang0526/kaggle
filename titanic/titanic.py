# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("./"))

# Any results you write to the current directory are saved as output.

dataset = pd.read_csv('./titanic/train.csv')
dataset_test = pd.read_csv('./titanic/test.csv')

X_index = list(range(2, 3)) + list(range(4, 8)) + list(range(9, 10)) + list(range(11, 12))
X = dataset.iloc[:, X_index].values
y = dataset.iloc[:, 1].values

# preprocessing

# nan

# label
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# gender
label_encoder_X = LabelEncoder()
X[:, 1] = label_encoder_X.fit_transform(X[:, 1])

# age 2

# embark 6

label_encoder_X_6 = LabelEncoder()
X[:, 6] = label_encoder_X_6.fit_transform(X[:, 6].astype(str))


