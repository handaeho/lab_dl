"""
ex11에서 저장한 Weight/bias pickle 파일을 읽어와 파라미터(가중치/편향 행렬)들을 화면에 출력
"""
import pickle

with open('Weight_bias.pickle', 'rb') as f:
    params = pickle.load(f)
    print(params)