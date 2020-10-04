
# Web_dev
<br>

웹 구축 과정을 정리 

<br>

## 일자 
- 2020년 06월 08일 월요일 

<br>

## 진행 상황

웹 진행상황은 아래와 같다. 

- 시스템 구현 : 69 % 


## 전체적인 구조 

```bash

deploy 

    └ deploy -> setting check

    └ member 
        └ views(함수)
            └ home 
            └ sign_up 
                - 회원 가입
            └ sign_in
                - 로그인 
            └ sign_out
            └ user_delete
            └ (change_info : 사용자 정보수정)

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
```

<br>
<br>

## 세부 구조

### member : 회원 관리 시스템
---

<br>
함수형태로 생성, 기호에 따라 클래스로도 생성 가능 
<br>

```py
# member/views.py

from django.shortcuts import render, redirect # URL의 변화여부가 필요하다면 Redirect
from django.http import HttpResponse, HttpResponseNotFound # HttpResponse : 응답에 대한 메타정보를 가지고 있는 객체
from django.views.decorators.csrf import csrf_exempt # csrf 공격방지 패키지

from django.contrib.auth.decorators import login_required # 로그인 
from django.contrib.auth import authenticate as auth
from django.contrib.auth import login as login
from django.contrib.auth import logout as logout
from django.contrib.auth import get_user_model

from django.db import connection # 디비연결을 위한 팩 

from .models import join # 모델 호출 
from models.error import * 
# cursor = connection.cursor()

User = get_user_model() # 변수 선언 - 디비

```
<br>
<br>

### 메인화면
---
<br>

#### 렌더(render) 
index에서 인자값으로 request을 받았어 -> 받았으니까 돌려줘야해        
그때 render로 뭔가를 돌려주는데 그게 html 페이지(위치값이랑 확장자명 빼먹지 말기 )     
<br>

```py 

@csrf_exempt
def home(request):
    '''
    메인 
    '''
    if request.method == 'GET':
        return render(request,'member/home.html')

```

<br>
<br>

### 회원가입 
---

#### request 
request = 사용자가 입력한 값과 기타 정보를 담고 있음 (GET, POST, request,등등..)     
GET을 요청했다면 요청받은 렌더링 값을 리턴     
    => 페이지 이동 클릭 개념으로 생각하면 수월      

```py 
@csrf_exempt
def sign_up(request):
    '''
    회원가입
    '''
    if request.method == 'GET':
        return render(request,'member/sign_up.html')


    elif request.method == 'POST':

        id = request.POST['username']
        pw = request.POST['password']
        
        user = User.objects.create_user(
            username = id,
            password = pw
            )
        user.save()
        return redirect('/')

```
<br>
<br>

### 로그인 
---

```py 
@csrf_exempt
def sign_in(request):
    '''
    로그인
    '''
    if request.method == 'GET':
        return render(request,'member/sign_in.html')

    elif request.method == 'POST':

        id = request.POST['username']
        pw = request.POST['password']
        
        # 디비 인증 
        user = auth(request, username=id, password=pw)
        # 세션설정
        if user is not None:
            login(request, user) # 세션 추가 
            print('====>','로그인성공' )
            print('userid: ', request.session['userid'])
            print('username:', request.session['username'])
            # 성공
            return redirect('/')
        else:
            """
            실패_ 에러 메시지 전송  {'error':"username or password is incorrect!"}
            """
            return HttpResponse(error)
```
<br>
<br>

### 유저로그아웃
---
```py 

@login_required
# sign_out - 로그아웃
def sign_out(request):
    """
    로그아웃
    """
    if request.method == 'GET' or request.method =='POST':
        logout(request) 
        return redirect('/')

```

<br>
<br>

### 유저 정보 삭제 
---
```py    
def user_delete(request):
    if request.method == "GET":
        rows = join.objects.all() # 데이터 베이스 객체를 날림
        rows.delete()
        print(rows)

        return redirect('/')
```
<br>

