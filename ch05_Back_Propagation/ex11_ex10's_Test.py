"""
ex10의 Two_Layer_Neural_Network Test
"""
import pickle

import numpy as np
import matplotlib.pyplot as plt

from ch05_Back_Propagation.ex10_MNIST_Two_Layer_NN_Propagation import TwoLayerNetwork
from dataset.mnist import load_mnist


if __name__ == '__main__':
    np.random.seed(106)

    # MNIST 데이터 로드
    (X_train, Y_train), (X_test, Y_test) = load_mnist(one_hot_label=True)

    # TwoLayerNetwork 클래스 객체 생성
    neural_net = TwoLayerNetwork(input_size=784, hidden_size=32, output_size=10)

    epochs = 50 # 학습 횟수
    batch_size = 128 # 한번에 학습 시키는 Input Data 개수
    learning_rate = 0.1 # learning_rate(학습률)

    # 반복할 크기 iter_size(학습이 한 번 완료될 주기, 한 번의 epoch)
    # = 전체 데이터 개수를 batch_size로 나누어, 한 번에 batch_size만큼 몇 번 학습할 것인가?
    iter_size = max(X_train.shape[0] // batch_size, 1)
    # ~> max(a, b): a, b 중 큰 값을 선택
    # 즉, 'X_train.shape[0] // batch_size'가 0이어도 최소한 1번은 학습하게 한다.
    print(iter_size)

    train_losses = [] # train_set의 loss의 변화 값이 저장될 리스트
    train_accuracies = [] # train_set의 accuracy 변화 값이 저장될 리스트
    test_accuracies = [] # test_set의 accuracy 변화 값이 저장될 리스트

    # 1회 epoch에서 iter_size만큼 반복 학습해 Weight / bias 행렬을 수정하고 loss 계산
    for i in range(iter_size) :
        # 학습 데이터를 랜덤하게 섞음(shuffle).
        # 인덱스 0 ~ 59,999를 랜덤하게 섞은 후, 섞인 인덱스로 X_train과 Y_train을 선택.
        idx = np.arange(len(X_train))  # [0, 1, 2, ..., 59999]
        np.random.shuffle(idx)

        # batch_size 개수만큼의 학습 데이터를 입력으로 해서 gradient 계산
        X_batch = X_train[idx[i * batch_size:(i+1) * batch_size]]
        Y_batch = Y_train[idx[i * batch_size:(i+1) * batch_size]]
        gradients = neural_net.gradient(X_batch, Y_batch)

        # 가중치/편향 행렬들을 수정
        for key in neural_net.params:
            neural_net.params[key] -= learning_rate * gradients[key]

    # loss를 계산해서 출력
    train_loss = neural_net.loss(X_train, Y_train)
    print('train_loss:', train_loss)

    # accuracy를 계산해서 출력
    train_acc = neural_net.accuracy(X_train, Y_train)
    print('train_acc:', train_acc)
    test_acc = neural_net.accuracy(X_test, Y_test)
    print('test_acc:', test_acc)

    # 위 과정을 epochs(100)회 반복 ------------------------------
    for x in range(epochs):
        # 학습 데이터를 랜덤하게 섞음(shuffle).
        # 인덱스 0 ~ 59,999를 랜덤하게 섞은 후, 섞인 인덱스로 X_train과 Y_train을 선택.
        idx = np.arange(len(X_train))  # [0, 1, 2, ..., 59999]
        np.random.shuffle(idx)

        for i in range(iter_size) :
            # batch_size 개수만큼의 학습 데이터를 입력으로 해서 gradient 계산
            X_batch = X_train[idx[i * batch_size:(i+1) * batch_size]]
            Y_batch = Y_train[idx[i * batch_size:(i+1) * batch_size]]
            gradients = neural_net.gradient(X_batch, Y_batch)

            # 가중치/편향 행렬들을 수정
            for key in neural_net.params:
                neural_net.params[key] -= learning_rate * gradients[key]

        # loss를 계산해서 train_losses 리스트에 추가하고 출력
        train_loss = neural_net.loss(X_train, Y_train)
        train_losses.append(train_loss)
        print(f'{x}번 train_loss:', train_loss)

        # train_accuracy를 계산해서 train_accuracies 리스트에 추가하고 출력
        train_acc = neural_net.accuracy(X_train, Y_train)
        train_accuracies.append(train_acc)
        print(f'{x}번 train_acc:', train_acc)

        # test_accuracy를 계산해서 test_accuracies 리스트에 추가하고 출력
        test_acc = neural_net.accuracy(X_test, Y_test)
        test_accuracies.append(test_acc)
        print(f'{x}번 test_acc:',test_acc)

    # epochs번 학습이 종료된 후, train_set의 loss / test_set의 accuracy list
    print(train_losses)
    print(len(train_losses))
    print(train_accuracies)
    print(len(train_accuracies))
    print(test_accuracies)
    print(len(test_accuracies))

    # epochs번 학습 종료 후, epochs-loss / epochs-accuracy 그래프 출력
    # epoch ~ loss 그래프
    x = range(epochs)
    plt.plot(x, train_losses)
    plt.title('Loss - Cross Entropy')
    plt.show()

    # epochs-accuracy 그래프
    plt.plot(x, train_accuracies, label='train_accuracy')
    plt.plot(x, test_accuracies, label='test_accuracy')
    plt.xlabel('EPOCH')
    plt.ylabel('ACCURACY')
    plt.title('ACCURACY')
    plt.legend()
    plt.show()

    # 신경망에서 학습이 모두 끝난 후, 파라미터(Weight/bias 행렬)를 pickle 파일로 저장
    with open('Weight_bias.pickle', 'wb') as f:
        pickle.dump(neural_net.params, f, pickle.HIGHEST_PROTOCOL)
