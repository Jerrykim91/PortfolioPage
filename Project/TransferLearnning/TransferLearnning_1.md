<br>

# Transfer Learnning : 전이 학습
<br>

Transfer Learnning의 장점
- 기본적인 성능 향상
- 전반적인 모델 개발 및 훈련 시간 단축, 그리고 우수한 모델
<br>
<br>

## 패키지 
---
<br>

```py
## 기본 

import numpy as np
import pandas as pd
import scipy as sp
from numpy.random import rand
pd.options.display.max_colwidth = 600

import matplotlib.pyplot as plt
%matplotlib inline
# 그래프 파라미터 설정
params = {'legend.fontsize': 'x-large',
          'figure.figsize': (15, 5),
          'axes.labelsize': 'x-large',
          'axes.titlesize':'x-large',
          'xtick.labelsize':'x-large',
          'ytick.labelsize':'x-large'}

plt.rcParams.update(params)

from IPython.display import display, HTML
import warnings
warnings.filterwarnings('ignore') # 워닝무시

## sklearn
from sklearn import preprocessing
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.model_selection import train_test_split


## 딥러닝
import tensorflow as tf
# 전처리 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# 아키텍쳐 - VGG16
from tensorflow.keras.applications import vgg16 as vgg
# 아키텍쳐 - 기본 요소 
from tensorflow.keras import optimizers,Model
from tensorflow.keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D,BatchNormalization
from tensorflow.python.keras.utils import np_utils

# 콜백 함수
from tensorflow.keras import callbacks
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping

```
<br>


## 준비 과정 
---
<br>


```py

BATCH_SIZE = 64        # 배치 사이즈 
EPOCHS = 100           # 에포크 
NUM_CLASSES = 4        # 학습할 카테고리 수 
LEARNING_RATE = 1e-4   # 러닝 레이트
MOMENTUM = 0.9

# 데이터 위치 경로 
PATH="C:\FoodClassification\data"

```

```py
base_model = vgg.VGG16(weights='imagenet', 
                       include_top=False, 
                       input_shape=(128, 128, 3))


# VGG16 모델의 세 번째 블록에서 마지막 층 추출
last = base_model.get_layer('block3_pool').output

```
<br>

## 모델 중비 
---

- 최상위층 없이 VGG16 로딩 
- 커스텀 분류기 준비 
- 모델의 맨위에 새로운 층 쌓기 -> ???? 

<br>

```py

# 최상위 층에 분류층 추가 

x = GlobalAveragePooling2D()(last)
x= BatchNormalization()(x)
x = Dense(256, activation='relu')(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.6)(x)
pred = Dense(NUM_CLASSES, activation='softmax')(x) # 학습 

# 원래 
model_basic = Model(base_model.input) + pred
# 궁금증 -> base_model.input ??? 무엇인고 ?

# 사용할 모델 
model = Model(base_model.input, pred)

# 모델 설계 확인 
model.summary()

```
<br>

## 체크포인트 생성 
---

콜백기능을 이용해서 손실값이 좋아질때 마다 학습된 모델을 저장한다.  
`verbose = 1` 를 이용하면 자동으로 메세지를 확인 할 수 있다. 

<br>

```py

checkpoint = keras.callbacks.ModelCheckpoint('model_cnn3-{epoch:03d}-{acc:03f}-{val_acc:03f}.h5', verbose=1,
                                             monitor='val_loss',
                                             save_best_only=True, mode='auto')

# 기능 

```
<br>

```py

# base_model의 레이어를 반복 
for layer in base_model.layers:
     layer.trainable = False  # -> 뭔지 모르겠음

```

