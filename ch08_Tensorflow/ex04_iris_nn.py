# -*- coding: utf-8 -*-
"""IRIS_NN.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/168rEEt-0PenrcCmABLDt8rBoY6vA6yKQ
"""

# Commented out IPython magic to ensure Python compatibility.
# Tensorflow 사용 버전 설정
# %tensorflow_version 2.x

"""**1. 로컬 PC에 저장된 CSV 파일을 Google Colab으로 upload**"""

from google.colab import files

uploaded = files.upload()

"""**2. CSV 파일을 읽고 데이터 프레임 생성**"""

import pandas as pd

df = pd.read_csv('./Iris.csv')

print(df.shape)
print(df)

"""**3. 데이터 전처리**"""

# 1) ID 컬럼 삭제
df = df.iloc[:, 1:6]
print(df)

# 2) pasndas.DataFrame -> numpy.ndarray로 변환
dataset = df.to_numpy()
print(type(dataset), dataset.shape)

# 3) 데이터(SL, SW, PL, PW)와 레이블(품종)을 분리
X = dataset[:, :-1].astype('float16')
Y = dataset[:, -1]
print(f'X: {X.shape}, Y: {Y.shape}')
print(X[:5])
print(Y[:5])

# 4) encoding 
# 레이블 데이터 타입을 문자열(setosa, versicolor, virginica)에서 숫자(0, 1, 2)로 변환
encoder = LabelEncoder()  # sklearn.preprocessing
encoder.fit(Y)
Y = encoder.transform(Y)
print(Y[:5])

# one-hot-encoding
Y = to_categorical(Y, 3, dtype='float16')  # tensorflow.keras.utils
print(Y[:5])

# 5) 학습 데이터 셋 / 테스트 데이터 셋 분할
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)

print(f'X_train: {X_train.shape}, Y_train: {Y_train.shape}')
print(f'X_test: {X_test.shape}, Y_test: {Y_test.shape}')
print(X_train[:5])
print(Y_train[:5])

"""**4. 신경망 모델 생성**"""

# 클래스와 모듈 import
import os
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras import Sequential
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# hidden_layer Dense(16, relu) -> output_layer Dense(3, softmax)
model = Sequential()

model.add(Dense(16, activation='relu', input_dim=4))

model.add(Dense(3, activation='softmax'))

"""**5. 신경망 학습**"""

# 신경망 모델 컴파일
model.compile(loss='categorical_crossentropy', 
              optimizer='adam', 
              metrics=['accuracy'])

# 일정 횟수동안 학습의 성과가 없으면 자동으로 stop
early_stop = EarlyStopping(monitor='val_loss', verbose=1, patience=10)

# 신경망 모델 학습 진행(fit)
history = model.fit(X_train, Y_train, batch_size=1, epochs=50,
          validation_data=(X_test, Y_test))

"""**6. 테스트 데이터로 평가**"""

eval = model.evaluate(X_test, Y_test)
print(f'X_test Loss: {eval[0]}, X_test Accuracy: {eval[1]}')

"""**7. Loss / Accuracy 그래프로 모델 성능 시각화**"""

# Loss/Accuracy vs Epoch plot
x = range(50)  # epoch
train_loss = history.history['loss']
test_loss = history.history['val_loss']
plt.plot(x, train_loss, c='blue', marker='.', label='Train loss')
plt.plot(x, test_loss, c='red', marker='.', label='Test loss')
plt.legend()
plt.show()

train_acc = history.history['accuracy']
test_acc = history.history['val_accuracy']
plt.plot(x, train_acc, c='blue', marker='.', label='Train accuracy')
plt.plot(x, test_acc, c='red', marker='.', label='Test accuracy')
plt.legend()
plt.show()

"""**8. confusion matrix & classification report**"""

from sklearn.metrics import confusion_matrix, classification_report

y_true = np.argmax(Y_test, axis=1)
print(y_true)

y_pred = np.argmax(model.predict(X_test), axis=1)
print(y_pred)

cm = confusion_matrix(y_true, y_pred)
print(cm)

report = classification_report(y_true, y_pred)
print(report)