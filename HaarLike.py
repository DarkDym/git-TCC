from tkinter import *
import numpy as np
from PIL import Image, ImageTk
import time
import cv2 as cv
from matplotlib import pyplot as plt
import os

from AbreIMG import abreImg as ai

#DIRA = "C://Users//Katy//Desktop//TCC Alleff//TCC_v1-3//CODES//IMAGENS//FRONTAL"

#Mudar PATH e DIRA caso seja mudado de computador, precisa ser o caminho absoluto
DIRA = "C://Users//Dymytry//Desktop//TCC Alleff//TCC Alleff//TCC_v1-3//CODES//IMAGENS//FRONTAL"
PATH = "C://Users//Dymytry//Desktop//TCC Alleff//TCC Alleff//TCC_v1-3//CODES//"

class harr:
    def __init__(self):
        pass
    def detecta(self,frame):
        #face_cascade = cv.CascadeClassifier('C://Users//Katy//Desktop//TCC Alleff//TCC_v1-3//CODES//haarcascade_frontalface_default.xml')
        face_cascade = cv.CascadeClassifier(PATH + 'haarcascade_frontalface_default.xml')
        #eye_cascade = cv.CascadeClassifier('C://Users//Katy//Desktop//TCC Alleff//TCC_v1-3//CODES//haarcascade_eye.xml')
        eye_cascade = cv.CascadeClassifier(PATH + 'haarcascade_eye.xml')

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        frame = cv.cvtColor(frame,cv.COLOR_RGB2BGR)

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        if type(faces) != tuple:
            for (x,y,w,h) in faces:
                frame = cv.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
                self.roi_gray = gray[y:y+h, x:x+w]
                self.roi_color = frame[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(self.roi_gray)
            #return self.roi_gray
            return frame
        else: 
            return frame
            #for (ex,ey,ew,eh) in eyes:
            #    cv.rectangle(self.roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
    def detecta2(self,frame):
        #face_cascade = cv.CascadeClassifier('C://Users//Katy//Desktop//TCC Alleff//TCC_v1-3//CODES//haarcascade_frontalface_default.xml')
        face_cascade = cv.CascadeClassifier(PATH + 'haarcascade_frontalface_default.xml')
        #eye_cascade = cv.CascadeClassifier('C://Users//Katy//Desktop//TCC Alleff//TCC_v1-3//CODES//haarcascade_eye.xml')
        eye_cascade = cv.CascadeClassifier(PATH + 'haarcascade_eye.xml')
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x,y,w,h) in faces:
            frame = cv.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
            self.roi_gray = gray[y:y+h, x:x+w]
            self.roi_color = frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(self.roi_gray)
        return self.roi_gray
    def cutFace(self):
        image = []
        for filePath in sorted(os.listdir(DIRA)):
            imagePath = os.path.join(DIRA,filePath)
            im = cv.imread(imagePath)
            image.append(im)
            imFlip = cv.flip(im,1)
            image.append(imFlip)
        lgt = len(image)
        print(lgt)
        os.chdir("./IMAGENS/FT")
        y = 1
        for x in range(0,lgt):
            aux = image[x]
            frame = self.detecta2(aux)
            frame = cv.resize(frame, (300,300))
            cv.imwrite(str(y)+"-11.jpg",frame)
            y = y + 1
        image.clear()

#root = harr()
#root.cutFace() 