# download data
# > kaggle competitions download -c facial-keypoints-detection -p ./facial_detection/


import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from set_current_dir import set_current_dir

import matplotlib.pyplot as plt


set_current_dir('facial_detection')

dataset = pd.read_csv('training.zip', compression='zip')
testset = pd.read_csv('test.zip', compression='zip')
idlookups = pd.read_csv('IdLookupTable.csv')

# a = list(map(int, dataset['Image'][0].split(' ')))
# max(a) = 253, min(a) = 2, Image is a 0-255 joined str

# [len(image.split(' ')) for image in dataset['Image']]
# all Image is 9216 size 96*96

# train_data.isnull().any().value_counts()
# True     28
# False     3
# dtype: int64
# so 28 columns missing values

dataset.fillna(method='ffill', inplace=True)

def getX(ds):
    l = [list(map(int, image.split(' '))) for image in ds['Image']]
    image_list = np.array(l, dtype='float')
    X = image_list.reshape(-1, 96, 96)
    return X


X_train = getX(dataset)

# dataset.describe()
# x/y max 96

y_train = dataset.iloc[:, :-1]


def show_image(images, features):
    plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(images[i])
        plt.grid(False)
        for j in range(15):
            x = features[i][j*2]
            y = features[i][j*2+1]
            plt.plot(x, y, 'xb')
    plt.show()


show_image(X_train, y_train.values)

model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=(96, 96)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(64, activation=tf.nn.relu),
    keras.layers.Dense(30)
])

model.compile(optimizer='adam',
              loss='mse',
              metrics=['mae', 'accuracy'])

history = model.fit(X_train, y_train, batch_size=512,
          epochs=40, verbose=1,
          validation_split=0.2)

X_test = getX(testset)

plt.imshow(X_test[0])
plt.show()

y_pred = model.predict(X_test)

show_image(X_test, y_pred)

# export pred

locations = []
for i in range(len(idlookups)):
    featureName = idlookups['FeatureName'][i]
    imageIndex = idlookups['ImageId'][i] - 1
    featureIndex = dataset.columns.get_loc(featureName)
    location = y_pred[imageIndex][featureIndex]
    locations.append(location)

