from tkinter import *
import numpy as np
from PIL import Image, ImageTk
import time
import cv2 as cv
from matplotlib import pyplot as plt

import gc

from AbreIMG import abreImg as ai

NEF = 10
#DIRA = "C://Users//Katy//Desktop//TCC Alleff//TCC_v1-3//CODES//IMAGENS//FT"
#Mudar DIRA caso seja mudado de computador, precisa ser o caminho absoluto
DIRA = "C://Users//Dymytry//Desktop//TCC Alleff//TCC Alleff//TCC_v1-3//CODES//IMAGENS//FT"
MAX_SLIDER_VALUE = 255
avgFace = 0
eface = []

def createNewFace(*args):        
    output = avgFace     
    for i in range(0, NEF):
        sValues = cv.getTrackbarPos("Weight" + str(i), "Trackbars");
        weight = sValues - MAX_SLIDER_VALUE/2
        output = np.add(output, eface[i] * weight)
    output = cv.resize(output, (0,0), fx=2, fy=2)
    cv.imshow("Result", output)
def resetSliderValues(*args):
    for i in range(0,NEF):
        cv.setTrackbarPos("Weight" + str(i), "Trackbars", int(MAX_SLIDER_VALUE/2))
    createNewFace()
def teste(aux):
    cv.namedWindow("TESTE",cv.WINDOW_AUTOSIZE)
    cv.imshow("TESTE",aux)
    cv.waitKey(0)
    cv.destroyAllWindows()

class eigenfaces:
    def __init__(self):
        pass
    def eigNovo(self):
        images = ai.openImg(self,DIRA)
        for i in range(0,len(images)):
            images[i] = cv.cvtColor(images[i],cv.COLOR_BGR2GRAY)
            print(images[i].shape)
        [gamma, qnt_gamma] = ai.createDataMatrix(self,images)
        print(gamma.shape)
    def classifica(self):
        images = ai.openImg(self,DIRA)
        sz = images[0].shape
        [data,qnt_data] = ai.createDataMatrix(self,images)
        images.clear()
        gc.collect()
        mean, eigenVec = cv.PCACompute(data, mean=None, maxComponents=qnt_data)
        global avgFace
        avgFace = mean.reshape(sz)

        #teste de face_media menos a face
        #teste(avgFace)
        images = ai.openImg(self,DIRA)
        print(len(images))
        for xi in images:
            img_teste = avgFace - xi
            #print("PRIMEIRA IMAGEM - " + str(img_teste.shape))
            img_teste2 = cv.cvtColor(img_teste,cv.COLOR_BGR2GRAY)
            teste(img_teste)
            teste(img_teste2)
            #print("SEGUNDA IMAGEM" + str(img_teste2.shape))
            cov_img_b = np.cov(img_teste2, rowvar=False, bias=True)
            #print(cov_img_b)
            VC, VA = np.linalg.eig(cov_img_b)
            print(VC)
            
            print(VA)
            lol = (300,300)
            VA = VA.reshape(lol)
            teste(VA)

            poar = img_teste2.dot(cov_img_b)
            print(poar)
            teste(poar)
        #teste de face_media menos a face
        
        phi = []
        phiT = []
        for i in range(0,qnt_data):
            data_aux = data[i].reshape(sz)
            phi_aux = np.subtract(data_aux,avgFace)
            #teste(phi_aux)
            phi.append(phi_aux)
            phiT.append(np.transpose(phi[i]))
        #teste(avgFace) PSI = (1/M) * SIGMA(data)
        A = []
        At = []
        C = []
        for x in range(0,qnt_data):
            A.append(phi[x])
            At.append(phiT[x])
            C.append(np.dot(At[x],A[x]))
        print(C)
        global eface
        eface = []
        print(len(eigenVec))
        for ev in eigenVec:
            #print(ev)
            eigen = ev.reshape(sz)
            #teste(eigen)
            #cv.namedWindow("TESTE",cv.WINDOW_AUTOSIZE)
            #cv.imshow("TESTE",eigen)
            #cv.waitKey(0)
            #cv.destroyAllWindows()
            #eigen = cv.resize(eigen,50,50)
            eface.append(eigen)
        cv.namedWindow("Result", cv.WINDOW_AUTOSIZE)
        output = cv.resize(avgFace, (0,0), fx=2, fy=2)
        cv.imshow("Result", output)
        cv.namedWindow("Trackbars", cv.WINDOW_AUTOSIZE)
        sliderValues = []
        for i in range(0, NEF):
            sliderValues.append(MAX_SLIDER_VALUE/2)
            cv.createTrackbar( "Weight" + str(i), "Trackbars", round(MAX_SLIDER_VALUE/2), MAX_SLIDER_VALUE, createNewFace)
        cv.setMouseCallback("Result", resetSliderValues)
        cv.waitKey(0)
        cv.destroyAllWindows()
    #def createNewFace(self,avg,sValues,eigenFaces):

tst = eigenfaces()
tst.eigNovo()
