
# 디렉토리 만들기 -> adr에 파일 이름만 추가 하면 됨 

def make_dir(adr):
    import os 

    """
    디렉토리 만들기 -> adr에 파일 이름만 추가 하면 됨 
    """
    # 현재 디렉토리 파일리스트 확인 
    base = os.getcwd()
    # full = base + adr
    tmp = adr.split('\\')

    for idx in range(len(tmp)):
        ck = base +'\\'+'\\'.join(tmp[:idx+1])
        if os.path.exists(ck) is False:
            os.mkdir(ck)
            print('데이터 확인 :', ck , os.path.isdir(ck))

# 만들려고 하는 

def func_dir(name):
    import os 

    # 만들고자 하는 dir 여부 확인 -> os.path.isdir()
    if not os.path.isdir(name):
        # dir 생성 
        os.makedirs(name)
        print(name, "폴더가 생성되었습니다.")
    else:
        print("해당 폴더가 이미 존재합니다.")