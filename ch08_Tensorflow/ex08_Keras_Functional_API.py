"""
Keras Functional API (교재 P.271 그림 8-11)
= 복잡한 모델을 생성할 수 있는 방식인 functional API(함수형 API)

앞서 케라스를 사용하여 모델을 설계하는 방식을 sequential API를 사용했다.
그런데 sequential API는 여러층을 공유하거나 다양한 종류의 입력과 출력을 사용하는 등의
복잡한 모델을 만드는 일을 하기에는 한계가 있다.

functional API는 각 층을 일종의 함수(function)로서 정의한다.
그리고 각 함수를 조합하기 위한 연산자들을 제공하는데, 이를 이용하여 신경망을 설계한다

<내용>
1) Input() 함수에 입력의 크기를 정의한다.

2) 필요한 layer 객체를 생성하고, 인스턴스를 호출한다.
   그리고 이전층을 다음층 함수의 입력으로 사용하고, 변수에 할당한다.

3) Model()의 instance를 생성하고 method에 입력과 출력을 정의한다.
   model로 저장하면 sequential API를 사용할 때와 마찬가지로 model.compile, model.fit 등을 사용 가능하다.

ex) encoder = Dense(128)(input)
    ~~~> 이 코드는 아래와 같이 두 개의 줄로 표현할 수 있다.
        encoder = Dense(128)
        encoder(input)
"""
import numpy as np

from tensorflow.keras import Sequential, Model
from tensorflow.keras import layers
from tensorflow.keras import Input


# x(n, 64) -> Dense(32) -> ReLU -> Dense(32) -> ReLU -> Dense(10) -> Softmax

# Sequential Ver.
seq_model = Sequential()

seq_model.add(layers.Dense(32, activation='relu', input_shape=(64, )))
seq_model.add(layers.Dense(32, activation='relu'))
seq_model.add(layers.Dense(10, activation='softmax'))

# model 요약 정보 출력
seq_model.summary()

print()

""" 참고로 모델을 생성하고, 컴파일을 한 뒤, 학습(fit)을 진행할 때, batch_size, epochs등을 설정하는 것임을 기억하자. """

# Functional API Ver.

# step 1) Input() 함수에 입력의 크기를 정의한다.
input_tensor = Input(shape=(64, )) # 입력 텐서의 shape 결정

# 2) 필요한 layer 객체를 생성하고, 인스턴스를 호출한다. 그리고 이전층을 다음층 함수의 입력으로 사용하고, 변수에 할당한다.
hidden1 = layers.Dense(32, activation='relu')(input_tensor)
hidden2 = layers.Dense(32, activation='relu')(hidden1)
output_tensor = layers.Dense(10, activation='softmax')(hidden2)

# 3) Model()의 instance를 생성하고 입력과 출력을 정의한다.
func_model = Model(inputs=input_tensor, outputs=output_tensor)

# model 요약 정보 출력
func_model.summary()

# 이제 만들어진 모델로 모델 compile -> 모델 fit -> 모델 evaluate 및 predict -> 모델 성능 개선 등 진행

# 모델 compile
func_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 모델 fit
np.random.seed(122)
x_train = np.random.random(size=(1000, 64))
y_train = np.random.randint(10, size=(1000, 10))

func_model.fit(x=x_train, y=y_train, batch_size=128, epochs=10, verbose=1)

# 모델 predict 및 evaluate
score = func_model.evaluate(x_train, y_train) # 연습이니까 train_set을 test_set으로
print('score =', score)
