
# Problem : 4차시 
<br>

진행중인 프로젝트에 관한 여러가지 이슈 그리고 해결방안을 포스팅하고자 한다. 

<br>

## 일자 
- 2020년 06월 02일 화요일 

<br>

## 진행 상황

현재 전체적인 진행상황은 아래와 같다. 

- 이미지 분석 : 45 % 
- 텍스트 분석 : 85 % (감소)
- 시스템 구현 : 10 % 

<br>
<br>


## 이슈 1 : 모델 설계 

<br>

우리가 가진 코드가 불필요한 층이 많고 정규화가 제대로 이루어 지지 않음 
그래서 층을 정리하고 정규화 코드를 추가하였음
그리고 전체적으로 코드가 벨런스가 맞지 않아서 여러가지 시도를 진행해봐야할것같다 

전체적으로 옵티마이저도 바꿔가면서 진행하는것이 필요 

- 이미지 사이즈 - 128로 고정 
- 배치사이즈 - 256, 64
- 드롭아웃 변경이 필요 0.3, 0.5  

<br>
<br>


## 이슈 2 : 시스템서버 설계

<br>

### 구조

- member
    - 회원 관리 시스템 
- service
    - 이미지 분석
    - 텍스트 분석

<br>

### 데이터 베이스 

- mongoDB 사용 

<br>

### 아웃라인 

- 본래 우리가 이미지를 인식해서 정답 데이터를 던져주면 
    텍스트 팀이 데이터를 받아서 해당 서비스를 제공  

- 산출물 : Web app

<br>
<br>

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

def sign_up():
    '''
    회원가입
    '''
    pass


def sign_in():
    '''
    로그인
    '''
    pass

 
def main():
    '''
    메인(로그인전)
    '''
    pass

# def change_info():pass

# 코드가 추가 퇼 수 있음 -> 핵심 기능일뿐 오해하지 말자 

```
<br>

### service 
---

```py

def text(food_name):
    """
    food_img에서 던져 주는 값을 제대로 아웃풋을 낼 것
    """

    return meau

# 임시 방편 
# img = '이미지'
def food_img(img):
    """
    당장은 test임
    """ 
    # 우리가 가진 데이터 리스트를 
    # 기반으로 이미지 랜덤 코드를 짜서 전달

    return food_name

def take_pic(click):
    """
    이미지 촬영 코드 
    """ 
    return img

# 코드가 추가 퇼 수 있음 -> 핵심 기능일뿐 오해하지 말자 

```

<br>
<br>


