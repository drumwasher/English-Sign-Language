#�ν�
import cv2
import mediapipe as mp
import numpy as np
import requests
import time

max_num_hands = 1
url='https://35fd-223-28-183-207.jp.ngrok.io/method'

s_flag=0 # �����÷���

gesture = {
    0:'a', 1:'b', 2:'c', 3:'d', 4:'e', 5:'f', 6:'g', 7:'h',
    8:'i', 9:'j', 10:'k', 11:'l', 12:'m', 13:'n', 14:'o', 
    15:'p', 16:'q', 17:'r', 18:'s', 19:'t', 20:'u', 21:'v',
    22:'w', 23:'x', 24:'y', 25:'z', 26:'space', 27:'clear', 28:'enter',29:'back',30:'state'
}

#���� ���� �׸���
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

#hand detection ��� �ʱ�ȭ
hands = mp_hands.Hands(
    max_num_hands = max_num_hands,
    min_detection_confidence = 0.5,
    min_tracking_confidence = 0.5)

#���۵�����(�հ��� ����) �н�
file = np.genfromtxt('testset.txt', delimiter=',')
angleFile = file[:,:-1]
labelFile = file[:,-1]
angle = angleFile.astype(np.float32)
label = labelFile.astype(np.float32)
knn = cv2.ml.KNearest_create()
knn.train(angle, cv2.ml.ROW_SAMPLE, label)

#��ķ
cap = cv2.VideoCapture(0)

#���� �ʱ�ȭ
startTime = time.time()
prev_index = 0
sentence = ''
recognizeDelay = 1

while True:
    ret, img = cap.read()
    if not ret:
        continue
    
    #��ó���� �̹��� 
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(imgRGB)
    
    #���� �νĵǸ� ����
    if result.multi_hand_landmarks is not None:
        for res in result.multi_hand_landmarks: 
            joint = np.zeros((21, 3))
            for j, lm in enumerate(res.landmark): #���� ������ ��ġ ����
                joint[j] = [lm.x, lm.y, lm.z]

            #joint��ȣ ���� ���
            v1 = joint[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19], :]
            v2 = joint[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], :]
            
            #������ ���� ���� ���ϱ�
            v = v2 - v1
            v = v / np.linalg.norm(v, axis=1)[:, np.newaxis] #nomallize
            compareV1 = v[[0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 16, 17], :]
            compareV2 = v[[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19], :]
            
            #acos���� �������ϱ�
            angle = np.arccos(np.einsum('nt,nt->n', compareV1, compareV2))
            angle = np.degrees(angle) #rdian�� degree������ ��ȯ
            
            #���� ����
            data = np.array([angle], dtype=np.float32)
            ret, results, neighbours, dist = knn.findNearest(data ,3)
            index = int(results[0][0])
            
            if index in gesture.keys():
                if index != prev_index:
                    startTime = time.time()
                    prev_index = index
                else:
                    if time.time() - startTime > recognizeDelay: #1�ʵ��� ���������̸�
                        if index == 26: #space
                            sentence += ' '
                        elif index == 27: #clear
                            sentence = '' 
                        elif index == 28: #enter
                            sentence = ''
                        elif index ==29: #back
                            sentence=sentence[:-1]
                        elif index ==30: #��� / ���� ���� ����
                            sentence = ''
                            s_flag+=1
                            s_flag%=2
                            if s_flag==0:
                                params={'stage':gesture[index]}
                                response=requests.get(url,params=params)
                        else: #a~z
                            if s_flag==1: #�����̸� ���ĺ� �߰�
                                sentence += gesture[index]
                        startTime = time.time()
                        #get���
                        if s_flag==1:
                            params={'stage':gesture[index]}
                            response=requests.get(url,params=params)

                            
                        
                #���� ��ġ�� ���� ����� ���ĺ� ���
                cv2.putText(img, gesture[index].upper(), (int(res.landmark[0].x * img.shape[1] - 10), int(res.landmark[0].y * img.shape[0] + 40)), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                    color=255, thickness=3)
                

            mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)
    #�ϴܿ� ���� ���
    cv2.putText(img, sentence, (20, 440), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                    color=255, thickness=3)
    
    #��ܿ� ��� / ���� �������
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