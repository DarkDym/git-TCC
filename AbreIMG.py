import sys
import os
import cv2 as cv
import numpy as np

class abreImg:
    def __init__(self):
        pass
    def openImg(self,path):
        image = []
        for filePath in sorted(os.listdir(path)):
            imagePath = os.path.join(path,filePath)
            im = cv.imread(imagePath)
            im = np.float32(im)/255.0
            image.append(im)
            imFlip = cv.flip(im,1)
            image.append(imFlip)
        return image
    def openImgray(self,path):
        image = []
        for filePath in sorted(os.listdir(path)):
            imagePath = os.path.join(path,filePath)
            im = cv.imread(imagePath,cv.IMREAD_GRAYSCALE)
            im = np.float32(im)/255.0
            image.append(im)
            imFlip = cv.flip(im,1)
            image.append(imFlip)
        return image
    def createDataMatrix(self,image):
        numImg = len(image)
        sz = image[0].shape
        data = np.zeros((numImg,sz[0] * sz[1]),dtype=np.float32)
        for i in range(0, numImg):
            img = image[i].flatten()
            data[i,:] = img
        image.clear()
        return data,numImg
    def cria(self,image):
        numImg = len(image)
        sz = image[0].shape
        data = np.zeros((numImg,sz[0] * sz[1] * sz[2]),dtype=np.uint8)
        for i in range(0, numImg):
            img = image[i].flatten()
            data[i,:] = img
        image.clear()    
        return data,numImg
    def saveImg(self,img_np,name,PATH):
        try:
            os.chdir(PATH)
            # print("VALOR DO IMG_NP: "+str(img_np))
            normimg = cv.normalize(img_np, None, alpha=0, beta=1,norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)
            normimg = normimg*255.0
            # print("VALOR DO TESTE: "+str(teste))
            # teste = np.array(img_np)
            cv.imwrite(name,normimg)
        except OSError:
            print("Caminho " + str(PATH) +" inválido.")
    def openImg2DB(self,path):
        image = []
        for filePath in sorted(os.listdir(path)):
            imagePath = os.path.join(path,filePath)
            im = cv.imread(imagePath)
            im = np.float32(im)/255.0
            image.append(im)
            # imFlip = cv.flip(im,1)
            # image.append(imFlip)
        return image