"""
Instance Call

클래스를 통해 생성된 객체. 이를 '인스턴스(instance)'라고 한다.
그리고 이렇게 생성된 객체에 특정 동작을 수행시키는 것을 '인스턴스 메소드 호출(instance method call)'이라 한다.

클래스의 'method'를 설정하면 메모리에 저장되고,
클래스의 'instance'를 생성하면 저장되어 있는 메모리의 주소를 참조하여 '__init__ method'를 읽어와, class 변수 값들을 가져온다.
instance는 단지 '메모리의 주소'를 가지고 있을뿐이지 'Call 할 수 있는 것은 아니다'.

ex) instance_name = class_name()
    instance_name.mathod_name ~~~> 가능(단지 class가 가진 메모리의 주소를 참조할 뿐)

    instance_name(100) ~~~> 불가능(instance에 '100'이라는 값을 주고 Call 불가)

그런데 Python에서는 '__call__ method'를 사용하면 instance_name을 method처럼 Call 할 수 있다.
왜나하면, '__call__ method'가 메모리에 저장된 '__init__ method'의 class 변수 값을 바꾸기 떄문.

단, class에서 '__call__ method'를 설정하지 않으면 instance를 call 할 수 없다.

ex) class에 '__call__ method'를 설정하고,
    instance_name = class_name()
    instance_name(100) ~~> 가능
"""


class Foo:
    def __init__(self, init_val=1):
        self.init_val = init_val


class Boo:
    def __init__(self, init_val=1):
        self.init_val = init_val

    def __call__(self, n):
        print('__call__ Call')
        self.init_val *= n

        return self.init_val


if __name__ == '__main__':
    # Foo Class의 instance 생성 - 생성자 호출
    foo = Foo()
    print('foo.init_val =', foo.init_val) # foo.init_val = 1

    print()

    # Boo Class의 instance 생성
    boo = Boo()
    print('boo.init_val =', boo.init_val) # boo.init_val = 1

    print()

    # 그런데 Python에서는 '__call__' method를 사용하면
    # instance_name을 method처럼 Call 할수 있다.
    boo(5)
    print('boo.init_val =', boo.init_val)
    # __call__ Call
    # boo.init_val = 5

    print()

    # callable: _call__ method를 구현한 instance
    print('foo callabel:', callable(foo)) # foo callabel: False ~~> class에 '__call__ method' 없음
    print('boo callabel:', callable(boo)) # boo callabel: True ~~~> class에 '__call__ method' 있음

    print()

    boo = Boo(2) # '__init__ method'에 있는 class 변수 값을 2로

    x = boo(2) # x = 4 ~> '__call__ method'를 call하고, 기능을 수행하여 '__init__ method'의 class 변수 값을 변경
    print('x = ', x)
    # __call__ Call
    # x =  4

    print()

    x = boo(x)
    print('x =', x)
    # __call__ Call
    # x = 16

    print()

    input = Boo(1)(5)
    print('input =', input)
    # __call__ Call
    # input = 5
    # ~> __call__ method를 call하고 __init__의 class 변수 값을 1로 하고, __call__ method에 5를 전달해 기능 수행
    # Boo(1): __init__ method를 call ---> (5): __call__ method를 call하고 5를 전달

    print()

    x = Boo(5)(input)
    print('x =', x)
    # __call__ Call
    # x = 25

    print()

    x = Boo(5)(x)
    print('x =', x)
    # __call__ Call
    # x = 125