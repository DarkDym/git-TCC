from tkinter import *
import numpy as np
from PIL import Image, ImageTk
import time
import cv2 as cv
from matplotlib import pyplot as plt
import json

import gc

import base64
from bson import Binary

from AbreIMG import abreImg as ai
from Database import db_mongo as db

NEF = 10
#DIRA = "C://Users//Katy//Desktop//TCC Alleff//TCC_v1-3//CODES//IMAGENS//FT"
#Mudar DIRA caso seja mudado de computador, precisa ser o caminho absoluto
DIRA = "C://Users//Dymytry//Desktop//TCC Alleff//TCC Alleff//TCC_v1-3//CODES//IMAGENS//FT"
MAX_SLIDER_VALUE = 255
avgFace = 0
eface = []
K = 50
TAM_IMG = (300,300)

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
def teste2(aux1,aux2):
    cv.namedWindow("TESTE",cv.WINDOW_AUTOSIZE)
    numpy_horizontal = np.hstack((aux1, aux2))
    cv.imshow("TESTE",numpy_horizontal)
    cv.waitKey(0)
    cv.destroyAllWindows()
    

class eigenfaces:
    def __init__(self):
        pass
    def img2db(self):
        client = db.conecta(self)
        images = ai.openImg(self,DIRA)
        for i in range(0,len(images)):
            images[i] = cv.cvtColor(images[i],cv.COLOR_BGR2GRAY)
        [gamma, qnt_gamma] = ai.createDataMatrix(self,images)
        print(len(images))
        for x in range(0,qnt_gamma):
            img_list = gamma[x].tolist()
            img_json = {"nome":"foto"+str(x+1),"pixel":img_list}

            # TESTE PARA SALVAR ARQUIVO E VER TAMANHO DELE
            # with open("teste2.json","w") as write_file:
                # json.dump(img_json,write_file,indent=4)
            # write_file.close()

            img_json = {"nome":"foto"+str(x+1),"pixel":Binary(gamma[x])}
            

            # stri = base64.b64encode(images[x])
            # with open("teste.txt","w") as salva:
                # salva.write(str(stri))
            # salva.close()
            # print(stri)
            # db.gamma2db(self,client,img_json)
            # break
            # FIM DO TESTE PARA SALVAR ARQUIVO E VER TAMANHO DELE

            # gamma_list = gamma[x].tolist()
            # gamma_json = {"nome":"foto"+str(x+1),"pixel":gamma_list}
            db.gamma2db(self,client,img_json)
        images.clear()
        # img_json.clear()
        db.kill_connection(self,client)
    
    def db2img(self):
        db_gamma = db.db2gamma(self)
        for x in db_gamma:
            print(x['nome'])
        db.kill_connection(self)
        
    
    def eigNovo(self):
        images = ai.openImg(self,DIRA)
        for i in range(0,len(images)):
            images[i] = cv.cvtColor(images[i],cv.COLOR_BGR2GRAY)
        [gamma, qnt_gamma] = ai.createDataMatrix(self,images)
        images.clear()
        gc.collect()
        #INICIO DO TESTE
        # print(gamma[0])
        # for x in range(0,qnt_gamma):
            # aux = np.array([gamma[x]])
            # aux_T = aux.transpose()
            # tst_json = gamma[x].tolist()
            # tst_json2 = {"nome":"foto"+str(x+1),"pixel":tst_json}
            # print(tst_json)
            # print(z)
            # y = json.dumps(gamma[x])
            # print(z)
            # db.gamma2db(self,tst_json2)
        # db.kill_connection()

        # print(aux)
        # print(aux.shape)
        # print(aux.transpose().shape)
        # print("INICIO DA MEDIA")
        #TERMINO DO TESTE
        
        psi = 0
        aux_psi = 0
        for x in range(0,qnt_gamma):
            aux_psi = gamma[x] + aux_psi
        psi = aux_psi/qnt_gamma
        phi = []
        for y in range(0,qnt_gamma):
            phi.append(gamma[y] - psi)
        # matrix_cov = 0
        # aux_cov = []
        phi_aux = np.array(phi)
        print("PHI_AUX SHAPE "+ str(phi_aux.shape))
        # print("PHI_AUX "+str(phi_aux))
        # aux_t = 0
        # for x in range(0,qnt_gamma):
            # aux_t = np.transpose(phi_aux[x])
            # print("TESTE: "+str(phi_aux[x].shape))
            # print("TESTE2: "+str(aux_t.shape))
            # print(np.dot(phi_aux[x],aux_t))
            # aux_cov.append(np.dot(phi_aux[x],aux_t))
            # break
        # print("AUX_COV SHAPE "+str(aux_cov[0].shape))
        # print("AUX_COV "+str(aux_cov))
        # aux_cov = np.array([aux_cov])
        # print("AUX_COV SHAPE "+str(aux_cov.shape))
        # matrix_cov = aux_cov/qnt_gamma
        # print("MATRIX_COV{SOMATORIO DE PHI} SHAPE: " + str(matrix_cov.shape))
        # print("MATRIX_COV{SOMATORIO DE PHI} VALOR: " + str(matrix_cov))
        A = np.array(phi)
        # print("SHAPE DO PHI_i: " + str(phi[0].shape))
        print(A.shape)
        A_T = A.transpose()
        print(A_T.shape)
        Cov = np.dot(A,A_T)
        print(Cov.shape)
        #print(Cov)
        [eg_vle,eg_vec] = np.linalg.eig(Cov)
        #EG_VEC é o vi do STEP 6.2

        print("VALOR DE EG_VEC" + str(eg_vec) + "SHAPE DO EG_VEC " + str(eg_vec.shape))
        print("VALOR DE EG_VLE" + str(eg_vle) + "SHAPE DO EG_VLE " + str(eg_vle.shape))
        
        # tst_phi = np.array(phi)
        # print("TST_PHI SHAPE "+str(tst_phi.shape))
        u = np.dot(A_T,eg_vec)
        # u_n = np.linalg.norm(u)
        # print(u_n)
        print("U shape: "+str(u.shape))
        print("PHI_AUX SHAPE: "+str(phi_aux.shape))
        u_t = u.transpose()
        print("U_T shape: "+str(u_t.shape))
        print("VALOR DE U_T[0]: "+str(u_t[0])+ " MAIS A NORMALIZAÇÂO DO VALOR "+str(np.linalg.norm(u_t[0])))
        w = np.dot(u_t,np.transpose(phi_aux))
        print("w SHAPE: "+str(w.shape))
        print("SHAPE DO w[0]: "+str(w[0].shape) + " | SHAPE DO u_t[0]: "+str(u[0].shape))
        u_mod = []
        for x in range(0,u_t.shape[0]):
            u_mod.append(np.linalg.norm(u_t[x]))
        print(len(u_mod))

        ord = False
        while not ord:
            ord = True
            for x in range(0,len(u_mod)-1):
                if (u_mod[x] < u_mod[x+1]):
                    u_mod[x],u_mod[x+1] = u_mod[x+1],u_mod[x]
                    u_t[x],u_t[x+1] = u_t[x+1],u_t[x]
                    ord = False
        print("MODIFICADO: "+str(u_mod))
        # tau = []        
        # for x in range(0,5):
            # aux_u = np.array([u[x]])
            # aux_w = np.array([w[x]])
            # aux_w = np.transpose(aux_w)
            # aux_u = np.transpose(aux_u)
            # print("TESTE DO AUX_U SHAPE: "+str(aux_u.shape))
            # print("TESTE DO AUX_U_T SHAPE: "+str(np.transpose(aux_u).shape))
            # break
            # aux = np.dot(w[x],aux_u.transpose())
            # print(aux.shape)
            # tau.append(np.dot(aux_w,aux_u))            
        # tau = np.array(tau)
        # print("TAU SHAPE: "+str(tau.shape))
        # print("TAU[0] SHAPE: "+str(tau[0].shape))
        

        # teste(u[0].reshape((300,300)))
        for x in range(0,K):
            teste2(phi_aux[x].reshape(TAM_IMG),u_t[x].reshape(TAM_IMG))
        

        # print("NORMALIZAÇÃO DO EG_VEC " + str(np.linalg.norm(eg_vec)))
        # u_i = np.dot(A_T,eg_vec)
        # print("u_i = A*v_i --> "+str(u_i))
        # print("u_i SHAPE: "+str(u_i.shape))
        # print("||u_i|| --> "+str(np.linalg.norm(u_i)))
        # ui_mod = []
        # for x in range(0,eg_vec.shape[0]):
            # for y in range(0,u_i.shape[0]):
                # if (eg_vec[x] == u_i[y]).all():
                    # ui_mod.append(u_i[y])
                    # break
        # print("SHAPE DO UI_MOD: "+str(ui_mod))
        # print("COMPARATIVO --> EG_VEC[0]: "+str(eg_vec[0])+ " VS U_i: "+str(u_i[0]))
        # u_i_best = []
        # for x in range(0,eg_vec.shape[0]):
            # u_i_best.append(u_i[x])
        
        # u_i_mod = sorted(u_i,reverse=True)
        # K_vet = []
        # for x in range(0,K):
            # K_vet.append(u_i_mod[x])
        # print("u_i ORDENADO: "+str(u_i_mod))
        # print("OS 50 MELHORES: "+str(K_vet))
        # [EGVALUE,EGVECTOR] = np.linalg.eig(matrix_cov)

        # print("VALOR DO AUTOVALOR" + str(EGVALUE) + "SHAPE DO AUTOVALOR " + str(EGVALUE.shape))
        # print("VALOR DE AUTOVETOR" + str(EGVECTOR) + "SHAPE DO AUTOVETOR " + str(EGVECTOR.shape))

        #INICIO TESTE DE REPRESENTAÇÃO DAS EIGENFACES
        
        #FIM TESTE DE REPRESENTAÇÃO DAS EIGENFACES

        eg_vle_T =np.array([eg_vec[0]]) 
        aux_phi = np.transpose(np.array([phi[0]]))
        w = np.dot(aux_phi,eg_vle_T)
        print("VALORES QUE ESTOU TESTANDO")
        eg_face = np.dot(w,eg_vec[0])
        if (eg_face == phi[0]).all():
            print("ESTA IGUAL")
        else:
            print("TA DIFERENTE")
        teste(phi[0].reshape((300,300)))
        teste(eg_face.reshape((300,300)))
        
        # w = []
        # for i in range(0,qnt_gamma):
        #     w.append(np.dot(eg_vle[i]))
        
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
# tst.db2img()
tst.eigNovo()
# tst.img2db()
