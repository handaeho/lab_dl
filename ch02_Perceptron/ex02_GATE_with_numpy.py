"""
numpy.ndarray를 사용한 게이트 구현
"""
import numpy as np


def test_perceptron(perceptron):
    """ x1, x2를 [0, 0] / [0, 1] / [1, 0] / [1, 1] 중 하나인 numpy.ndarray 타입으로 생성"""
    for x1 in (0, 1):
        for x2 in (0, 1):
            x = np.array([x1, x2])
            result = perceptron(x)
            print(f'{x1, x2} -> {result}')

    return x


def and_gate(x):
    """ test_perceptron()의 리턴 값을 x로 받아 AND GATE 계산 """
    # w = [w1, w2]인 numpy.ndarray 타입
    w = np.array([1, 1])
    b = -1  # 편향
    y = np.sum(w * x) + b
    if y > 0:
        return 1
    else:
        return 0


def nand_gate(x):
    """ test_perceptron()의 리턴 값을 x로 받아 NAND GATE 계산 """
    # w = [w1, w2]인 numpy.ndarray 타입
    w = np.array([-1, -1])
    b = 1  # 편향
    y = np.sum(w * x) + b
    if y >= 0 :
        return 1
    else :
        return 0

def or_gate(x):
    """ test_perceptron()의 리턴 값을 x로 받아 OR GATE 계산 """
    # w = [w1, w2]인 numpy.ndarray 타입
    w = np.array([1, 1])
    b = -1  # 편향
    y = np.sum(w * x) + b
    if y >= 0:
        return 1
    else :
        return 0


if __name__ == '__main__':
    # 1) AND GATE
    test_perceptron(and_gate)

    print(' ======================================================================= ')

    # 2) NAND GATE
    test_perceptron(nand_gate)

    print(' ======================================================================= ')

    # 3) OR GATE
    test_perceptron(or_gate)
