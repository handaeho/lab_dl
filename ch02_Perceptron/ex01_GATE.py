"""
Perceptron(퍼셉트론) : 다수의 신호를 입력받아 하나의 신호를 출력하는 구조

input (x1, x2, ...) -> f(x) -> output (x1w1 + x2w2 + ...) ===> y = wx + b (w는 weight(가중치), b는 bias(편향))

1) AND 게이트 : x1, x2에서 둘 다 모두 1이면 1, 하나라도 0이면 0
2) NAND 게이트 : Not AND. 'AND 게이트'의 결과와 반대
3) OR 게이트 : x1, x2에서 둘 중 하나라도 1이면 1
4) XOR 게이트 : 배타적 논리 합. x1, x2에서 둘 중 하나만 1일때 1, 둘 다 0이거나 1이면 0
"""
# 1) AND GATE
def and_gate(x1, x2):
    """
    x1, x2에서 둘 다 모두 1이면 1, 하나라도 0이면 0
    """
    w1, w2 = 1, 1  # 가중치
    b = -1  # 편향
    y = (x1 * w1) + (x2 * w2) + b  # y = wx + b
    if y > 0: # 가중치를 곱한 입력의 총합(y)이 0보다 크다면 1을, 아니라면 0을 리턴
        return 1
    else:
        return 0


# 2) NAND GATE
def nand_gate(x1, x2):
    """
    Not AND. 'AND 게이트'의 결과와 반대
    """
    w1, w2 = -1, -1  # 가중치
    b = 1  # 편향
    y = (x1 * w1) + (x2 * w2) + b  # y = wx + b
    if y < 0: # AND GATE의 결과와 반대로 y가 0보다 작으면 0을, 아니라면 1을 리턴
        return 0
    else:
        return 1

# 3) OR GATE
def or_gate(x1, x2):
    """
    x1, x2에서 둘 중 하나라도 1이면 1
    """
    w1, w2 = 1, 1  # 가중치
    b = 1  # 편향
    y = (x1 * w1) + (x2 * w2) + b  # y = wx + b
    if y > 1 : # 가중치를 곱한 입력의 총합(y)이 1보다 크다면 1을, 아니라면 0을 리턴
        return 1
    else :
        return 0


# 4) XOR GATE
def xor_gate(x1, x2):
    """
    배타적 논리 합. x1, x2에서 둘 중 하나만 1일때 1, 둘 다 0이거나 1이면 0
    XOR GATE는 NAND와 OR의 결과를 AND한 결과로 구현된다.
    """
    s1 = nand_gate(x1, x2)
    s2 = or_gate(x1, x2)
    y = and_gate(s1, s2)

    return y


if __name__ == '__main__':
    # 1) AND GATE
    for x1 in (0, 1) :
        for x2 in (0, 1) :
            print(f'{x1, x2} -> {and_gate(x1, x2)}')
    print(' ======================================================================= ')
    # 2) NAND GATE
    for x1 in (0, 1) :
        for x2 in (0, 1) :
            print(f'{x1, x2} -> {nand_gate(x1, x2)}')
    print(' ======================================================================= ')
    # 3) OR GATE
    for x1 in (0, 1) :
        for x2 in (0, 1) :
            print(f'{x1, x2} -> {or_gate(x1, x2)}')
    print(' ======================================================================= ')
    # 4) XOR GATE
    for x1 in (0, 1) :
        for x2 in (0, 1) :
            print(f'{x1, x2} -> {xor_gate(x1, x2)}')




