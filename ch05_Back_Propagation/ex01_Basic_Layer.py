"""
Back Propagation(역전파)

Computational Graph(계산 그래프) : 복수개의 노드와 엣지로 계산 과정을 자료구조 형태의 그래프로 표현한 것.

f(x)= x^n일 때, f'(x) = df/dx = nx^n-1이다. 이 미분 계산을 그래프 자료구조 형태로 나타내면,
    'x-> [미분] -> df/dx'와 같다.
이때, 출발점부터 종착점까지 순서대로 진행되는 것을 'Forward Propagation(순전파)',
반대로 종착점부터 출발점으로 진행되는 것을 'Back Propagation(역전파)'라고 한다.

전체 계산이 아무리 복잡하고 변수가 많아도 각 단계에서 노드가 하는 일은 '국소적 계산'이다.
이는 단순하지만 그 결과를 다음 노드에 전달해가며 전체를 구성하기 때문에 복잡한 계산을 할 수 있게 한다.

예를 들어, '100원짜리 사과 2개를 샀고 소비세 10%가 붙은 최종 가격을 구하는 계산'은
'사과 100원 -> [*2] -> [*1.1] -> 최종 금액'이 된다.
이때 '최종 금액을 구하는 계산'이 'Forward Propagation(순전파)'가 되는것이며,
만약 '사과 가격이 오르면 최종 금액이 어떻게 변하는지' 알고싶을 때, 이는 '사과 가격에 대한 지불 금액의 미분'으로 표현 가능하고
이를 구하는것이 'Back Propagation(역전파)'이다.

또한 같은 방법으로 이렇게 사과 금액에 대한 미분 뿐만 아닌, 개수에 대한 미분이나 소비세에 대한 미분등으로 각 요소의 영향도 알 수 있다.
그리고 중간까지 구한 미분에 대한 결과를 공유 할 수 있어서 다수의 미분을 효율적으로 계산 할 수 있다.

이처럼 계산 그래프의 이점은 '순전파와 역전파'를 활용해서 '각 변수의 미분을 효율적으로 계산'할 수 있는 것이다.

정리하자면 'Forward Propagation'은 '시작점부터 각 노드와 엣지의 상태에 따라 순서대로 계산되어 결과가 출력되는 계산'이며,
'Back Propagation'은 '반대의 방향으로 진행되는 계산'으로 '미분을 통해 전 단계가 지금 단계의 노드에 어떤 영향'을 미쳤는지 알 수 있다.

y = f(x)의 계산 그래프 x -> [f] -> y에서
역전파의 계산 순서 : E * dy/dx <- [f] <- E
                  신호 E에 노드의 국소적 미분(dy/dx)를 곱하고, 다음 노드로 전달.

- 합성 함수 : 여러 함수로 구성된 함수.
             z = t^2(단, t=x+y) ~~~> z = t^2 = (x+y)^2
             합성 함수의 미분은 합성 함수를 구성하는 각 함수의 미분의 곱으로 나타낼 수 있다.

- 연쇄 법칙 : 합성 함수의 원리를 이용해 z=t^2(단, t=x+y)일 때,
             dz/dx = dz/dt * dt/dx, dz/dt = 2t이고, dt/dx = 1이므로
             dz/dx = dz/dt * dt/dx = 2t * 1 = 2(x+y)
"""
import numpy as np


class MultiplyLayer:
    """ 100원짜리 사과 2개를 사고, 10%의 소비세가 붙은 최종 가격을 구해보자."""
    def __init__(self):
        self.x = None
        self.y = None

    def forward(self, x, y):
        """ Forward Propagation(순방향 전파) """
        self.x = x
        self.y = y

        return x * y

    def backward(self, delta_out):
        """ Backword Propagation(역방향 전파) """
        # 원래는 입력이 x, 출력이 y지만, 역전파이므로 x와 y를 바꾸어서 y를 입력으로, x를 출력으로 바꾼다.
        dx = delta_out * self.y
        dy = delta_out * self.x

        return dx, dy


class AddLayer:
    def __init__(self):
        pass

    def forward(self, x, y):
        return x + y

    def backward(self, dout):
        dx, dy = dout, dout
        return dx, dy


if __name__ == '__main__':
    # MultiplyLayer 객체 생성
    apple_layer = MultiplyLayer()

    # Forward Propagation(순방향 전파)
    apple = 100 # 사과 가격
    n = 2 # 개수

    # 사과 2개의 총 가격 계산
    apple_price = apple_layer.forward(apple, n) # 순방향 전파
    print('사과 2개의 가격 =', apple_price)
    # 사과 2개의 가격 = 200

    # tax_layer를 MultiplyLayer 객체로 생성
    tax_layer = MultiplyLayer()

    # 'tax=1.1'로 설정해서 구매시 세금이 포함된 최종 가격 계산
    tax = 1.1
    total_price = tax_layer.forward(apple_price, tax)
    print('세금이 포함된 사과 2개의 최종 가격 =', total_price)
    # 세금이 포함된 사과 2개의 최종 가격 = 220.00000000000003

    # f = a * n * t 라고 할 때,
    # tax가 1 증가하면 전체 가격은 얼마가 증가? -> df/dt
    # 사과 개수가 1 증가하면 전체 가격은 얼마가 증가? -> df/dn
    # 사과 가격이 1 증가하면 전체 가격은 얼마가 증가? -> df/da

    # Backword Propagation(역방향 전파) ~> 역전파에서는 '각 순전파의 출력에 대한 미분값'을 인수로 받는다.
    delta = 1.0 # 가장 처음 역전파 될 값

    dprice, dtax = tax_layer.backward(delta)
    print('dprice =', dprice)
    print('dtax =', dtax)  # df/dt: tax 변화에 대한 전체 가격 변화율

    dapple, dn = apple_layer.backward(dprice)
    print('dapple =', dapple)  # df/da: 사과 단가 변화에 대한 전체 가격 변화율
    print('dn =', dn)  # df/dn: 사과 개수 변화에 대한 전체 가격 변화율

    # AddLayer 테스트
    add_layer = AddLayer()
    x = 100
    y = 200
    dout = 1.5
    f = add_layer.forward(x, y)
    print('f =', f) # f = x + y
    dx, dy = add_layer.backward(dout) # df/dx = 1, df/dy = 1
    print('dx =', dx) # dx * dout
    print('dy =', dy) # dy * dout





