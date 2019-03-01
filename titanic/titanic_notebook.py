
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

os.chdir(os.path.join(os.getcwd(), 'titanic'))

train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

combine = [train_df, test_df]

print(train_df.columns.values)

train_df.head()

train_df.tail()

train_df.info()

print('_'*40)

test_df.info()

train_df.describe()

train_df.describe(include=['O'])

train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)

train_df[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)

train_df[['SibSp', 'Survived']].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)

train_df[['Parch', 'Survived']].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)