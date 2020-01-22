"""
>> GoogLeNet - (교재 P.271 그림 8-10)
= layer를 하나씩 쌓아 올려 순차적으로 통과 하지 않고,
layer가 수평적으로 놓여 있고, 이 여러 layer를 하나의 데이터에 한꺼번에 적용해서 그 결과를 결합 후, 다음 구조로 보낸다.
이 구조를 'Inception'이라 하며, 'Inception'구조를 하나의 구성 요소로 여러개를 결합하여 사용하는 것이 GoogLeNet의 특징이다.
단, 같은 층에 수평적으로 놓인 layer는 서로 연결 되지 않는다.

================================================================================================================

>> ResNet(Residual Network) - (교재 P.272 그림 8-12)
= 학습할 때 층이 지나치게 깊으면 학습이 잘 되지 않고, 오히려 성능이 떨어진다.
ResNet에서는 이 문제를 해결하기 위해 '스킵 연결'을 도입한다. 층의 깊이에 비례해 성능 향상을 기대할 수 있다.

# 스킵 연결 ~> 입력 데이터를 Convolution_layer를 건너 뛰고 바로 Output에 더하는 구조.
              입력 x를 연속한 두 Conv_layer를 건너 뛰고 출력에 바로 연결한다.
              이 단축 경로가 없다면 Output은 'F(x)'가 되지만, 스킵연결로 인해 Output은 'F(x) + x'가 된다.
              스킵 연결은 층이 깊어져도 학습을 효율적으로 할수 있게 해주는데,
              이는 Back Propagation 때 스킵 연결이 신호 감쇠를 막아주기 때문이다.

스킵 연결은 입력 데이터를 '그대로'흘리는 것으로, 역전파 때도 상류의 기울기를 그래도 하류로 보낸다.
핵심은 상류의 기울기에 아무런 수정도 하지 않고 '그대로' 흘리는 것이다.
스킵 연결로 기울기가 작아지거나 지나치게 커질 걱정 없이 앞 층에서의 '의미있는 기울기'를 하류로 전할수 있다.

ResNet은 Conv_layer를 2개 층마다 건너뛰면서 층을 깊게 한다.

# 전이 학습 ~> 학습된 가중치(또는 그 일부)를 다른 신경망에 복사한 후, 그대로 재학습을 수행한다.
              예를 들어, VGG와 구성이 같은 신경망을 준비하고 미리 학습된 가중치를 초기값으로 설정한 후,
              새로운 데이터 셋으로 재학습을 수행한다.
              전이 학습은 '보유한 데이터 셋이 적을 때 유용'한 방법이다.

여기서는 'GoogLeNet', 'ResNet'의 구조 그대로를 구현 하지는 않고, 'ex08_Functional_API'에 이어 같은 구조를 구현해 보자.

1) GoogLeNet
Input_tensor(784, ) -> Dense(64) -> ReLU -> Dense(32) -> ReLU -> Dense(10) -> Output_tensor(10, ) -> Softmax
                    -> Dense(64) -> ReLU ->
2) ResNet
Input_tensor(784, ) -> Dense(32) -> ReLU -> [건너뛰고 Dense(32) -> ReLU -> Dense(32) ->  ReLU]
                                                -> F(x) + x -> Output_tensor(10, ) -> Softmax

"""
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, Add, concatenate


# GoogLeNet ===========================================

# Input_tensor 생성
input_tensor = Input(shape=(784,))

# Input_tensor가 한 층에 수평적으로 놓인 두 개의 hidden_layer(Dense layer)를 각각 통과
x1 = Dense(units=64, activation='relu')(input_tensor)
x2 = Dense(units=64, activation='relu')(input_tensor)

# 두 개의 hidden_layer(Dense layer)를 통과한 두 결과를 연결
concat = concatenate([x1, x2])

# 그 다음 hidden_layer에 연결된 결과 전달
x = Dense(32, activation='relu')(concat)

# Output_tensor
output_tensor = Dense(10, activation='softmax')(x)

# 모델 생성
model = Model(input_tensor, output_tensor)

# 모델 정보 요약
model.summary()


# ResNet ===========================================

# input_tensor 생성
input_tensor = Input(shape=(784, ))

# hidden_layer(Dense)
fx = Dense(units=32, activation='relu')(input_tensor)
x = Dense(units=32, activation='relu')(fx)
x = Dense(units=32, activation='relu')(x)

# 입력데이터 x를 출력에 더해준다.(F(x) + x)
x = Add()([x, fx])

# F(x) + x를 output_tensor에 전달
output_tensor = Dense(10, activation='softmax')(x)

# 모델 생성
model = Model(input_tensor, output_tensor)

# 모델 정보 요약
model.summary()