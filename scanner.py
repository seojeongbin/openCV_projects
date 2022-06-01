import sys
import numpy as np
import cv2

# ========== param ==========
color1 = (192, 192, 255)
color2 = (128, 128, 255)
r_size = 25
# ========== param ==========

def drawROI(img, corners):
    cpy = img.copy()

    for pt in corners:
        cv2.circle(cpy, tuple(pt.astype(int)), r_size, color1, -1, cv2.LINE_AA) # draw circle filled color

    cv2.line(cpy, tuple(corners[0].astype(int)), tuple(corners[1].astype(int)), color2, 2, cv2.LINE_AA)
    cv2.line(cpy, tuple(corners[1].astype(int)), tuple(corners[2].astype(int)), color2, 2, cv2.LINE_AA)
    cv2.line(cpy, tuple(corners[2].astype(int)), tuple(corners[3].astype(int)), color2, 2, cv2.LINE_AA)
    cv2.line(cpy, tuple(corners[3].astype(int)), tuple(corners[0].astype(int)), color2, 2, cv2.LINE_AA)

    display = cv2.addWeighted(img, 0.3, cpy, 0.7, 0) 
    # 기존 이미지와 직사각형그린 cpy를 합친다. 가중치 다르게 줌으로써 뒤에 배경 보이도록.
    # 그냥 return cpy해도 되긴함. 연산과정이 없어져서 속도는 더 빠름. 하지만 뒤에 배경 안보임.
    return display


def onMouse(event, x, y, flags, param):
    global srcQuad, dragSrc, ptOld, src

    # 1. 마우스 눌린경우 : 원안에 들어온게 하나라도 있으면 그 지점 True로 바꾸고 break
    if event == cv2.EVENT_LBUTTONDOWN: 
        for i in range(4):
            if cv2.norm(srcQuad[i] - (x, y)) < r_size: # 마우스 지점(x,y)이 srcQuand안쪽에 들어와있는경우 / 25 : 위에서 반지름을 25라고 지정해줬으므로
                dragSrc[i] = True # 드래그 허용
                ptOld = (x, y)
                break

    # 2. 마우스 떼면 싹 다 false
    if event == cv2.EVENT_LBUTTONUP:
        for i in range(4):
            dragSrc[i] = False

    # 3. 마우스 움직이는 경우
    if event == cv2.EVENT_MOUSEMOVE:
        for i in range(4):
            if dragSrc[i]: # 그냥 마우스가 움직이고 있는경우가 아닌, 눌린경우에 한해서 체크하기위해 True인 경우만
                dx = x - ptOld[0]
                dy = y - ptOld[1]

                srcQuad[i] += (dx, dy) # 꼭짓점 갱신해준다

                cpy = drawROI(src, srcQuad) # 갱신한 점으로 다시 이어준다
                cv2.imshow('img', cpy)
                ptOld = (x, y)
                break

if __name__ == "__main__" : 

    src = cv2.imread('./scanned_test.jpg')

    if src is None:
        print('Image open failed!')
        sys.exit()

    h, w = src.shape[:2]
    dw = 500 # 출력할영상의 가로크기
    dh = round(dw * 297 / 210)  # 출력영상 세로크기 (a4 비율적용)

    srcQuad = np.array([[30, 30], [30, h-30], [w-30, h-30], [w-30, 30]], np.float32) # 일단 점들 임의로 잡은거임
    dstQuad = np.array([[0, 0], [0, dh-1], [dw-1, dh-1], [dw-1, 0]], np.float32)
    dragSrc = [False, False, False, False]

    ## 초기 그리기 전 최초 화면 생성해주는 기능
    display = drawROI(src, srcQuad)

    cv2.imshow('img', display)
    cv2.setMouseCallback('img', onMouse)

    while True:
        key = cv2.waitKey()
        if key == 13:  # ENTER 키 : break되면서 다음 과정들이 진행됨
            break
        elif key == 27:  # ESC 키 : 전부 다 닫힘
            cv2.destroyWindow('img')
            sys.exit()

    # 투시 변환
    pers = cv2.getPerspectiveTransform(srcQuad, dstQuad)
    dst = cv2.warpPerspective(src, pers, (dw, dh), flags=cv2.INTER_CUBIC)

    # 결과 영상 출력
    cv2.imshow('dst', dst)
    cv2.waitKey()
    cv2.destroyAllWindows()
