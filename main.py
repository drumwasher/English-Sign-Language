#인식
import cv2
import mediapipe as mp
import numpy as np
import requests
import time

max_num_hands = 1
url='https://35fd-223-28-183-207.jp.ngrok.io/method'

s_flag=0 # 상태플래그

gesture = {
    0:'a', 1:'b', 2:'c', 3:'d', 4:'e', 5:'f', 6:'g', 7:'h',
    8:'i', 9:'j', 10:'k', 11:'l', 12:'m', 13:'n', 14:'o', 
    15:'p', 16:'q', 17:'r', 18:'s', 19:'t', 20:'u', 21:'v',
    22:'w', 23:'x', 24:'y', 25:'z', 26:'space', 27:'clear', 28:'enter',29:'back',30:'state'
}

#손의 뼈대 그리기
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

#hand detection 모듈 초기화
hands = mp_hands.Hands(
    max_num_hands = max_num_hands,
    min_detection_confidence = 0.5,
    min_tracking_confidence = 0.5)

#동작데이터(손가락 각도) 학습
file = np.genfromtxt('testset.txt', delimiter=',')
angleFile = file[:,:-1]
labelFile = file[:,-1]
angle = angleFile.astype(np.float32)
label = labelFile.astype(np.float32)
knn = cv2.ml.KNearest_create()
knn.train(angle, cv2.ml.ROW_SAMPLE, label)

#웹캠
cap = cv2.VideoCapture(0)

#변수 초기화
startTime = time.time()
prev_index = 0
sentence = ''
recognizeDelay = 1

while True:
    ret, img = cap.read()
    if not ret:
        continue
    
    #전처리된 이미지 
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(imgRGB)
    
    #손이 인식되면 실행
    if result.multi_hand_landmarks is not None:
        for res in result.multi_hand_landmarks: 
            joint = np.zeros((21, 3))
            for j, lm in enumerate(res.landmark): #손의 점들의 위치 저장
                joint[j] = [lm.x, lm.y, lm.z]

            #joint번호 각도 계산
            v1 = joint[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19], :]
            v2 = joint[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], :]
            
            #관절의 대한 벡터 구하기
            v = v2 - v1
            v = v / np.linalg.norm(v, axis=1)[:, np.newaxis] #nomallize
            compareV1 = v[[0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 16, 17], :]
            compareV2 = v[[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19], :]
            
            #acos으로 각도구하기
            angle = np.arccos(np.einsum('nt,nt->n', compareV1, compareV2))
            angle = np.degrees(angle) #rdian을 degree값으로 변환
            
            #동작 참조
            data = np.array([angle], dtype=np.float32)
            ret, results, neighbours, dist = knn.findNearest(data ,3)
            index = int(results[0][0])
            
            if index in gesture.keys():
                if index != prev_index:
                    startTime = time.time()
                    prev_index = index
                else:
                    if time.time() - startTime > recognizeDelay: #1초동안 같은문자이면
                        if index == 26: #space
                            sentence += ' '
                        elif index == 27: #clear
                            sentence = '' 
                        elif index == 28: #enter
                            sentence = ''
                        elif index ==29: #back
                            sentence=sentence[:-1]
                        elif index ==30: #대기 / 실행 상태 변경
                            sentence = ''
                            s_flag+=1
                            s_flag%=2
                            if s_flag==0:
                                params={'stage':gesture[index]}
                                response=requests.get(url,params=params)
                        else: #a~z
                            if s_flag==1: #실행이면 알파벳 추가
                                sentence += gesture[index]
                        startTime = time.time()
                        #get통신
                        if s_flag==1:
                            params={'stage':gesture[index]}
                            response=requests.get(url,params=params)

                            
                        
                #손의 위치에 현재 모션의 알파벳 출력
                cv2.putText(img, gesture[index].upper(), (int(res.landmark[0].x * img.shape[1] - 10), int(res.landmark[0].y * img.shape[0] + 40)), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                    color=255, thickness=3)
                

            mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)
    #하단에 문장 출력
    cv2.putText(img, sentence, (20, 440), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                    color=255, thickness=3)
    
    #상단에 대기 / 실행 문장출력
    if s_flag==1:
        cv2.putText(img, 'Running', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                    color=255, thickness=3)
    else:
        cv2.putText(img, 'Waiting', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                    color=255, thickness=3)

    cv2.imshow('HandTracking', img)
    cv2.waitKey(1)
    if cv2.waitKey(1) == ord('q'):
        break
#cap.release()