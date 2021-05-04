#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np

path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
path2 = cv2.data.haarcascades + "haarcascade_eye.xml"

# Inicializa o classificador cascade
face_classifier = cv2.CascadeClassifier(path) 
olhos_classifier = cv2.CascadeClassifier(path2) 

# configura a captura de imagem da webcam
vc=cv2.VideoCapture(0)

# se a webcam abrir pego um frame
if vc.isOpened():
    rval, frame = vc.read()
else:
    rval = False

while rval:
    # Converte o frame para escala de cinza 
    img1_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  
    # Realiza a detecção de face na imagem em cinza
    faces_return = face_classifier.detectMultiScale(img1_gray, scaleFactor = 1.2, minNeighbors = 5)
    # Faz a varredura na lista de faces detectadas em faces_return
    for (x,y,w,h) in faces_return:
        # Desenha um retangulo em cada face detectada
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        # Faz um crop da face detectada em cinza
        roi_gray = img1_gray[y:y+h, x:x+w]
        # Faz o mesmo crop da face com a imagem colorida
        roi_color = frame[y:y+h, x:x+w]
        # Realiza a detecção dos olhos na imagem da face em cinza
        eyes = olhos_classifier.detectMultiScale(roi_gray)
        # Faz a varredura na lista de olhos detectados eyes
        for (ex,ey,ew,eh) in eyes:
            # Desenha um retangulo em cada olho detectado
            # Note o seguinte, estamos desenhando o retangulo sobre o crop da face colorido. 
            # Alguns alunos desenharam sobre "frame" e notaram que os retangulos ficavam na posição errada. 
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
        

    # Exibe saida da imagem
    cv2.imshow("resultado", frame)
    # Atualiza com um novo frame
    rval, frame = vc.read()

    # ESC para sair do programa
    key = cv2.waitKey(10)
    if key == 27:
        break

vc.release()
cv2.destroyAllWindows()
 