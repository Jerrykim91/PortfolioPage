# Building powerful image classification models using very little data
# https://keraskorea.github.io//posts/2018-10-24-little_data_powerful_model/


# 수천 수백장정도의 작은 데이터셋을 가지고도 강력한 성능을 낼수 있는 모델을 만들어 볼것 
# 세가지 방법을 이용해 진행 

# 작은 네트워크를 처음부터 학습 (앞으로 사용할 방법의 평가 기준)
# 기존 네트워크의 병목특징 (bottleneck feature) 사용 ***
# 기존 네트워크의 상단부 레이어 fine-tuning

"""
ImageDataGenerator: 실시간 이미지 증가 (augmentation)
flow              : ImageDataGenerator 디버깅
fit_generator     : ImageDataGenerator를 이용한 케라스 모델 학습
Sequential        : 케라스 내 모델 설계 인터페이스
keras.applications: 케라스에서 기존 이미지 분류 네트워크 불러오기
레이어 동결 (freezing)을 이용한 fine-tuning

학습 데이터: 각각 1000장의 강아지 고양이 사진 사용 
검증 데이터: 각각 400장

"""
# 캐글 데이터를 이용해 진행 ->  필요 패키지 : 케라스, SciPy, PIL

# "“deep learning is only relevant when you have a huge amount of data"
# '컴퓨터가 알아서' 데이터의 특성을 파악하려면 학습할 수 있는 데이터가 많아야 한다.
# 딥러닝의 장점 -> 재사용성 

# 데이터 전처리와 Augmentation
# 이미지를 사용할 때마다 임의로 변형을 가함으로써 
# 마치 훨씬 더 많은 이미지를 보고 공부하는 것과 같은 학습 효과를 얻는것

# 과적합 (overfitting), 즉 모델이 학습 데이터에만 맞춰지는 것을 방지
# 새로운 이미지도 잘 분류할 수 있게 한다. 

"""
ImageDataGenerator 
- 전처리 과정을 돕기 위해서 케라스 
    ->  ImageDataGenerator 를 제공 
    - 학습도중에 이미지에 임의 변형 및 정규화 적용 
    - generator를 생성
        - flow(data, labels), flow_from_directory(directory) 두 가지 함수를 사용
        - fit_generator, evaluate_generator 함수를 이용
            - generator로 이미지를 불러와서 모델을 학습


"""

# Let's start! : train_set 2000pic

#  ImageDataGenerator 
from keras.preprocessing.image import ImageDataGenerator # 이미지 augmentation을 도와주는 친구

# 이미지를 마음대로 바꾸면 곤란하니까 저희가 몇 가지 지시 사항
"""
rotation_range            : 이미지 회전 범위 (degrees)
width_shift, height_shift : 그림을 수평 또는 수직으로 랜덤하게 평행 이동시키는 범위 (원본 가로, 세로 길이에 대한 비율 값)
rescale                   : 원본 영상은 0-255의 RGB 계수로 구성되는데, 
                            이 같은 입력값은 모델을 효과적으로 학습시키기에 너무 높습니다 (통상적인 learning rate를 사용할 경우). 
                            그래서 이를 1/255로 스케일링하여 0-1 범위로 변환시켜줍니다. 
                            이는 다른 전처리 과정에 앞서 가장 먼저 적용됩니다.

shear_range               : 임의 전단 변환 (shearing transformation) 범위
zoom_range                : 임의 확대/축소 범위
horizontal_flip           : True로 설정할 경우, 50% 확률로 이미지를 수평으로 뒤집습니다.
                            원본 이미지에 수평 비대칭성이 없을 때 효과적입니다. 
                            즉, 뒤집어도 자연스러울 때 사용하면 좋습니다.

fill_mode 이미지를 회전, 이동하거나 축소할 때 생기는 공간을 채우는 방식
그외에 다양한 인자들이 존재 위 인자는 대표적인 아이들  
"""
datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

img = load_img('data/train/cats/cat.0.jpg')  # PIL 이미지로드
x = img_to_array(img)           # (3, 150, 150) 크기의 NumPy 배열
x = x.reshape((1,) + x.shape)   # (1, 3, 150, 150) 크기의 NumPy 배열

# 아래 .flow() 함수는 임의 변환된 이미지를 배치 단위로 생성
# 지정된 `preview/` 폴더에 저장
i = 0
for batch in datagen.flow(x, batch_size=1,
                          save_to_dir='preview', save_prefix='cat', save_format='jpeg'):
    i += 1
    if i > 20:
        break  # 이미지 20장을 생성

# 잘못된 학습 예시일 경우 augmentation 인자를 컨트롤해서 수정 

# ------------------------------------------------------------------- #
"""
# 작은 CNN 밑바닥부터 학습하기 - 코드 40줄, 정확도 80%
    - 이미지 분류에 적합한 모델은 CNN
    - 데이터가 충분하지 않기 때문에 과적합 문제를 우선적으로 해결 
    - 이미지를 효과적으로 분류하려면 대상의 핵심 특징에 집중!!!

# Generator를 이용해서 이미지 augmentation을 수행하는 목적이 바로 이런 과적합을 막는 데에 있다.
단순 기하학적인 변형으로 이미지의 특성이 크게 달라지진 않는다.
가장 중요한 것은 모델의 엔트로피 용량, 
    - 즉, 모델이 담을 수 있는 정보의 양을 줄이는 것

모델의 복잡도를 늘리면 담을 수 있는 정보가 많아져서 이미지에서 더 많은 특징을 파악이 가능해진다.
그만큼 반대되는 방해 요소들이 생길것이다. 

만약 모델의 복잡도를 줄여 중요한 몇 가지 특징에 집중해서 분류 대상을 정확히 파악한다면
새로운 이미지도 정확하게 구분해 낼 수 있게 된다. 

# 엔트로피 용량을 조절하는 방법은 다양하다. 
    - 대표적인 파라미터 개수를 조절하는 방법 -> 레이어 개수와 레이어 크기
    - L1, L2 정규화 (regularization) 같은 가중치 정규화 기법 -> 학습하면서 모든 가중치를 반복적으로 축소하는 방법
        - 결과적으로 핵심적인 특징에 대한 가중치만 남게 되는 효과

# 레이어 개수, 그리고 레이어당 필터 개수가 적은 소규모 CNN을 학습
    - 데이터 augmentation 및 드롭아웃 (dropout) 기법을 사용
    - 드롭아웃은 과적합을 방지할 수 있는 또 다른 방법
    - 하나의 레이어가 이전 레이어로부터 같은 입력을 두번 이상 받지 못하도록 하여 데이터 augmentation과 비슷한 효과를 가진다. 
    
# Convolution 레이어 3개를 쌓아놓은 심플한 형태
    - ReLU 활성화 함수를 사용
    - max-pooling을 적용
"""