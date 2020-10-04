
# Web_dev
<br>

웹 구축 과정을 정리 

<br>

## 일자 
- 2020년 06월 03일 수요일 

<br>

## 진행 상황

웹 진행상황은 아래와 같다. 

- 시스템 구현 : 20 % 


## 전체적인 구조 

```bash

deploy 

    └ deploy -> setting check

    └ member 
        └ views(함수)
            └ main 
            └ sign_up 
                - 회원 가입
            └ sign_in
                - 로그인 
            └ (change_info : 사용자 정보수정)

    └ service
        └ views (함수)
            └ food_img 
            └ text 
            └ take_pic  

    └ static
        └ js  : 미정
        └ css : 미정

    └ templates

        └ member 
            └ main.html : before login / after login
                - 첫화면 
                    - 리콰이얼 멤버 -> 다르게 show

            └ sign_up.html 
                - 성공 -> login page
                - ( 실패-> 팝업창 + 로그인 페이지 )

            └ sign_in.html 
                - 성공 -> main
                - 실패 -> 실패 팝업창 + 로그인 페이지2( 회원가입 권유, 계정찾기)
            └ (change_info : 옵션)

        └ service
            └ base.html : after login
                - 로그인 한사람만 이페이지를 리턴 받는다.
                - 이미지 인식을 위한 버튼 클릭 

            └ retrun_info.html
                - 인식된 이미지 정보를 보여줌 
                - 추천 관련 서비스 제공 
```

<br>
<br>

## 세부 구조

### member : 회원 관리 시스템
---

```py 

def sign_up(request):
    '''
    회원가입
    '''
    if request.method == 'GET':
        return render(request,'member/sign_up.html')
    elif request.method == 'POST':
        # id = request.POST['username']
        # pw = request.POST['password']
        pass


def sign_in(request):
    '''
    로그인
    '''
    if request.method == 'GET':
        return render(request,'member/sign_in.html')
    elif request.method == 'POST':
        # id = request.POST['username']
        # pw = request.POST['password']
        pass
 

def main(request):
    '''
    메인 
    '''
    if request.method == 'GET':
        return render(request,'member/main.html')
    

# def change_info():pass

```
<br>

### service 
---

```py

import numpy as np
import logging

from PIL import Image
from threading import Thread
from django.http import HttpResponse
from django.shortcuts import render, redirect
from django.views.decorators.csrf import csrf_exempt
from django.db import connection

from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
# from keras.preprocessing.image import image, ImageDataGenerator
# from keras.models import load_model


# logger = logging.getLogger(__name__)
# cursor = connection.cursor()
# model =load_model('.\\food_keras_model.h5')
# # Create your views here.
# print( '모델을 올린다 => ', type( model) )

# 텍스트 서비스 
def text(food_name):
    """
    food_img에서 던져 주는 값을 제대로 아웃풋을 낼 것
    """
    meau = '메뉴'
    return meau


# 음식 모델
@csrf_exempt
def food_img(request):
    """
    음식 모델
    """ 
    pass
    # if request.method == 'GET':
    #         print(1)
    #         return render(request, 'service/service.html') 

    # elif request.method=='POST':
        
    #     print('123456')
    #     logger.debug('uplaod calll')
    #     print(request)
        
    #     img = request.FILES['img']

    #     #img = ContentFile(img.read())
    #     #img = image.(PATH, target_size=(224,224))
        
    #     path = default_storage.save('img.jpg', ContentFile(img.read()))
    #     print('.\\'+path)
    #     img = image.load_img('.\\'+path, target_size=(224,224))
    #     #print(img.size)# 이미지는 잘 읽어 와짐
    #     img_t = image.img_to_array(img)
    #     print('111')
    #     img_t = np.expand_dims(img_t, axis=0)
    #     print('222')
    #     img_num = img_t.astype('float32')/255.
        
    #     # PATH='C:\\Users\\admin\\Desktop\\편집\\김치찌개\\Img_119_0085.jpg'
    #     # pic = Image.open(PATH)
    #     # img = image.load_img(PATH, target_size=(224,224))
    #     # img_t = image.img_to_array(img)
    #     # img_t = np.expand_dims(img_t, axis=0)
    #     # img_num = img_t.astype('float32')/255.
    #     # print(img_num.size)

    #     print( '모델을 올린다 => ', type( model) )
    #     pred = model.predict_classes(img_num)
    #     print(pred)
        
        

    #     return render(request, 'service/service.html') #redirect('/member/join1.html')

# 사진 촬영
def take_pic(click):
    """
    이미지 촬영 코드 
    """ 
    img ='이미지 촬영 데이터 '

    return img


```
