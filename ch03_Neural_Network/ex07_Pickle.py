"""
Pickle Data Type

- Serialize(직렬화) : picking. 객체 타입 -> 파일 저장
- Deserialize(역 직렬화) : unpicking. 파일 -> 객체 타입
"""
import pickle

arr = [1, 100, 'A', 3.141592]

# 객체(obj)를 파일(f)에 저장 ~> Serialize(직렬화)
with open('array.pickle', mode='wb') as f: # w: wirte, b: binary
    pickle.dump(arr, f)

# 파일(f)을 객체(obj)로 생성 ~> Deserialize(역 직렬화)
with open('array.pickle', mode='rb') as f: # r: read. b: binary
    data = pickle.load(f)

print(data)

# -----------------------------------------------------------------------------

data = {
    'name': '오쌤',
    'age': 16,
    'k1': [1, 2.0, 'AB'],
    'k2': {'tel': '010-0000-0000', 'email': 'jake@test.com'}
}

# data 객체를 data.pkl 파일에 저장: serialization
with open('data.pkl', mode='wb') as file:
    pickle.dump(data, file)

# data.pkl 파일을 읽어서 딕셔너리 객체를 복원: deserialization
with open('data.pkl', mode='rb') as file:
    restored = pickle.load(file)
print(restored)


