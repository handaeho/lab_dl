"""
교재 p.160 그림 5-15의 빈칸 채우기.

apple = 100원, n_a = 2개
orange = 150원, n_o = 3개
tax = 1.1라고 할 때,

전체 과일 구매 금액을 AddLayer와 MultiplyLayer를 사용해서 계산하세요.
df/dapple, df/dn_a, df/dorange, df/dn_o, df/dtax 값들도 각각 계산하세요.
"""
from ch05_Back_Propagation.ex01_Basic_Layer import MultiplyLayer, AddLayer

apple, n_a = 100, 2
orange, n_o = 150, 3
tax = 1.1

apple_mul = MultiplyLayer()  # 뉴런 생성
apple_price = apple_mul.forward(apple, n_a)  # forward propagation
print('apple_price =', apple_price)

orange_mul = MultiplyLayer()  # 뉴런 생성
orange_price = orange_mul.forward(orange, n_o)  # forward propagation
print('orange_price =', orange_price)

add_gate = AddLayer() # (사과 단가 * 개수) + (오렌지 단가 * 개수)
price = add_gate.forward(apple_price, orange_price)
print('price =', price) # 과일 총 가격

tax_mul = MultiplyLayer() # 과일 총 가격 * 세금
total_price = tax_mul.forward(price, tax)
print('total_price =', total_price) # 지불할 총액

# 역전파(backward propagation, back-propagation)
dprice, dtax = tax_mul.backward(1)
print('dprice =', dprice)
print('dtax =', dtax)  # df/dtax

dapple_price, dorange_price = add_gate.backward(dprice)
print('dapple_price =', dapple_price)
print('dorange_price =', dorange_price)

dapple, dn_a = apple_mul.backward(dapple_price)
print('dapple =', dapple)  # df/dapple
print('dn_a =', dn_a)  # df/dn_a

dorange, dn_o = orange_mul.backward(dorange_price)
print('dorange =', dorange)  # df/dorange
print('dn_o =', dn_o)  # df/dn_o
