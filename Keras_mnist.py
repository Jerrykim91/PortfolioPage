import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models, layers, backend
from tensorflow.keras.datasets import mnist  # datasets

# 데이터 셋 준비 
(X_train, y_train), (X_test, y_test) = mnist.load_data()
img_rows, img_cols = X_train.shape[1:] # 이미지 크기 
# print(img_rows, img_cols)   # 28, 28

# 데이터의 포멧에 따라, 채널의 위치를 조정
backend.image_data_format()

# 데이터를 이런 형식으로 맟춰서 준비 

# 채널 => 흑백이미지(1), 칼라(3) => (60000, 28, 28, 3)
input_channel = 1  

if backend.image_data_format == 'channels_last':
    # (1, rows, cols) => (1, H, W)
    # X_train.reshape(빈자리알아서(-1),채널,img_rows, img_cols)
    X_train = X_train.reshape( -1, 1, img_rows, img_cols )  # (60000, 1, 28, 28)
    X_test  = X_test.reshape( -1, 1, img_rows, img_cols )   # (10000, 1, 28, 28)
    input_shape = ( input_channel, img_rows, img_cols )
    
else :  # 'channels_last'
    X_train = X_train.reshape( -1, img_rows, img_cols, 1 )
    X_test  = X_test.reshape( -1, img_rows, img_cols, 1 )
    input_shape = ( img_rows, img_cols ,input_channel )

# print(X_train.shape, X_test.shape)
# print(X_train.dtype)# 데이터 성분, 범위, 타입

# 타입 변경 
X_train = X_train.astype('float32')
X_test  = X_test.astype('float32')
print(X_train.dtype)# 데이터 성분, 범위, 타입


# 정규화
# float32로 변화 처리하지만, 의미는 다르다.
print(y_train[:10])

# 텐서플로우에서 사용한 방식으로 사용해 본다면 
# 원-핫 인코딩 스타일로 작성해 본다면 케라스의 유틸리티를 사용 
# 답을 구성하는 클래스의 총수( 가짓수, 답안의 케이스 수)
num_classes = 10
y_train = keras.utils.to_categorical(y_train, num_classes)
print(y_train[:10])

# 동일하게 테스트셋에도 
y_test = keras.utils.to_categorical(y_test, num_classes)

"""
# 정규화 표준화 

- 기본적인 튜토리얼은 각 픽셀값이 0과 1로 정규화 되어있다. 
- 정규화 과정을 거치지 않은 상태로
    이파일은 픽셀이 0과 255사이의 값을 가진다.
    그렇기 때문에 전체 픽셀값을 255로 나누어서 정규화를 진행 -> 더욱 빠르게 연산이 가능하다.
"""

# 정규화 처리(최대값 기준) 
# -> 이부분 잘모르겠다. 텐서 문법인가...? 
X_train /= 255
X_test  /= 255


# print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

# 신경망 구성하기 
# 인공신경망을 만든다고 선언적 표현, 모델 생성
model = models.Sequential()

# 합성곱층 생성 추가
# filters : 출력 채널수 -> 1을 넣어서 32가 출력
# kernal_size : 3x3
model.add( layers.Conv2D(32, 
                         kernel_size=(3,3),
                         strides=(1, 1),
                         padding='valid',
                         activation='relu',
                         input_shape=input_shape
                         ) )

# 풀링층 생성 추가 
model.add(layers.MaxPool2D(pool_size=(2,2),))
# 과적합 방지 추가 
model.add(layers.Dropout(0.25))

# 합성곱층 생성 추가 : 기본값들이 있는 부분들은 생략해서 표현
model.add(layers.Conv2D(64, 
                         kernel_size=(3,3),
                         #strides=(1, 1),
                         #padding='valid',
                         activation='relu',
                         #input_shape=input_shape,
                         ))


# 풀링층 생성 추가
model.add( layers.MaxPool2D( pool_size=(2,2) ) )

# 과적합 방지 추가
model.add( layers.Dropout(0.25) )
# 전결합층 추가 
model.add(layers.Dense(0.5))

# 출력층 추가 => y
# num_classes => 10
model.add( layers.Dense( num_classes, activation='softmax' ) )
# 손실함수, 최적화, 정확도 추가 혹은 컴파일
model.compile( loss=keras.losses.categorical_crossentropy,
               optimizer='rmsprop',
               metrics=['accuracy']
              )

# 훈련 및 테스트 

# 실험값, 임계값
# 실험 횟수, 세대수, 훈련 횟수
epochs     = 10 # 설정값
# 1회 학습시 사용되는 데이터양
batch_size = 128 # 설정값

model.fit( X_train, y_train, 
          epochs     = epochs,
          batch_size = batch_size,
          validation_split=0.2          
          )

# 테스트 데이터로 확인
score = model.evaluate(X_test, y_test)
print(score)