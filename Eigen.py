from tkinter import *
import numpy as np
from PIL import Image, ImageTk
import time
import cv2 as cv
from matplotlib import pyplot as plt
import json
import time

import gc

import base64
from bson import Binary

from AbreIMG import abreImg as ai
from Database import db_mongo as db
from Google_Drive import connect_drive as gd

NEF = 10
#Mudar DIRA caso seja mudado de computador, precisa ser o caminho absoluto
DIRA = "C://Users//Dymytry//Desktop//TCC Alleff//git-TCC//IMAGENS//FT"
DIRA_MOD = "C://Users//Dymytry//Desktop//TCC Alleff//git-TCC//IMAGENS//TESTE_FT"
DIRA_TESTE = "C://Users//Dymytry//Desktop//TCC Alleff//git-TCC//IMAGENS//TESTE2_FT"
DIRA_TESTE2 = "C://Users//Dymytry//Desktop//TCC Alleff//git-TCC//IMAGENS//TESTE3_FT"
PATH_SAVE = "C://Users//Dymytry//Desktop//TCC Alleff//git-TCC//IMAGENS//EIGEN"
PATH_METRI = "C://Users//Dymytry//Desktop//TCC Alleff//git-TCC//METRICAS//"
MAX_SLIDER_VALUE = 255
avgFace = 0
# eface = []
K = 70
E_PHI = 35
# THETA = 
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
def teste3(aux1,aux2,aux3):
    cv.namedWindow("TESTE",cv.WINDOW_AUTOSIZE)
    numpy_horizontal = np.hstack((aux1, aux2, aux3))
    cv.imshow("TESTE",numpy_horizontal)
    cv.waitKey(0)
    cv.destroyAllWindows()


class eigenfaces:
    def __init__(self):
        pass

    def show_img(self,aux):
        cv.namedWindow("TESTE",cv.WINDOW_AUTOSIZE)
        cv.imshow("TESTE",aux)
        cv.waitKey(0)
        cv.destroyAllWindows()

    def testa_psi(self):
        client = db.conecta(self)
        time_start = time.time()
        psi_mod = db.get_psiM(self,client)
        time_end = time.time()
        for data in psi_mod:
            psi_id = data["Id_drive"]
            print(psi_id)
        GA = gd.connect()
        total_time = time_end - time_start
        print("TEMPO TOTAL DECORRIDO DO ACESSO AO MONGO(1 ARQUIVOS): EM Segundos:"+str(total_time)+" Em Minutos: "+str(total_time/60))
        time_start = time.time()
        data_drive = gd.get_teste(GA,psi_id)
        time_end = time.time()
        total_time = time_end - time_start
        print("TEMPO TOTAL DECORRIDO DO ACESSO AO DRIVE(1 ARQUIVOS): EM Segundos:"+str(total_time)+" Em Minutos: "+str(total_time/60))
        # print(data_drive["PSI"])

    def teste_cov(self):
        client = db.conecta(self)
        time_start = time.time()
        cov_mod = db.get_covM(self,client)
        time_end = time.time()
        cov_id = []
        for data in cov_mod:
            cov_id.append(data["Id_drive"])
        GA = gd.connect()
        data_drive = []
        total_time = time_end - time_start
        print("TEMPO TOTAL DECORRIDO DO ACESSO AO MONGO(70 ARQUIVOS): EM Segundos:"+str(total_time)+" Em Minutos: "+str(total_time/60))
        print("ACABEI DE PEGAR DO MONGO.")
        time_start = time.time()
        for x in range(0,len(cov_id)):
            data_drive_aux = gd.get_teste(GA,cov_id[x])
            data_drive.append(data_drive_aux["COV"])
            # print(data_drive["COV"])
            print("ACABEI DE PEGAR O PRIMEIRO ARQUIVO - "+str(x))
        time_end = time.time()
        total_time = time_end - time_start
        print("TEMPO TOTAL DECORRIDO DO ACESSO AO DRIVE(70 ARQUIVOS): EM Segundos:"+str(total_time)+" Em Minutos: "+str(total_time/60))

    def teste_phi(self):
        client = db.conecta(self)
        time_start = time.time()
        phi_mod = db.get_phiM(self,client)
        time_end = time.time()
        phi_id = []
        for data in phi_mod:
            phi_id.append(data["Id_drive"])
        GA = gd.connect()
        data_drive = []
        total_time = time_end - time_start
        print("TEMPO TOTAL DECORRIDO DO ACESSO AO MONGO(196 ARQUIVOS): EM Segundos:"+str(total_time)+" Em Minutos: "+str(total_time/60))
        print("ACABEI DE PEGAR DO MONGO.")
        time_start = time.time()
        for x in range(0,len(phi_id)):
            data_drive_aux = gd.get_teste(GA,phi_id[x])
            data_drive.append(data_drive_aux["PHI"])
            # print(data_drive["COV"])
            print("ACABEI DE PEGAR O PRIMEIRO ARQUIVO - "+str(x))
        time_end = time.time()
        total_time = time_end - time_start
        print("TEMPO TOTAL DECORRIDO DO ACESSO AO DRIVE(196 ARQUIVOS): EM Segundos:"+str(total_time)+" Em Minutos: "+str(total_time/60))

    def get_omegaface(self,gamma_frm):
        # Função que recebe a face cropada da Tela_principal a partir do HaarLike, a imagem já vem cortada no valor de TAM_IMG (Obs: pode ser BGR ou GRAY)
        # necessário calcular os omegas e phi da imagem recebida, para cálculo euclidiano.
        frm_mod = np.resize(gamma_frm,TAM_IMG)
        sz = frm_mod.shape
        data = np.zeros(sz[0] * sz[1],dtype=np.float32)
        gamma_flat = frm_mod.flatten()
        
        client = db.conecta()
        GA = gd.connect()
        psi = gd.get_psi(GA)
        phi_mod = gamma_flat - psi["PSI"]

        #   Preciso pegar as covariancia do drive
        cov = []
        for z in range(0,K):
            cov_aux = gd.get_cov(GA,z)
            cov.append(cov_aux["COV"])
        cov = np.array(cov)
        print("SÓ PRA SABER O SHAPE DE COV PEGO DO DRIVE: "+str(cov.shape))
        #   Preciso pegar as covariancia do drive

        # omega_json = []
        # flag_rec = []
        # s_project = []
        # s_erk = []
        # w_aux = 0
        # omega_project = []
        # for x in range(0,K):
        #     w = np.dot(u_cov[x],phi_mod[y])
        #     w_aux = (w*u_cov[x]) + w_aux
        #     omega_project.append(w)
        # omega_project = np.array(omega_project)
        # s_project.append(np.std(omega_project))
        # phi_project = w_aux
        # erk = []
        # err_phi = []
        # for x in range(0,phi_aux.shape[0]):
        #     aux_erk2 = np.linalg.norm(omega_project - omega[x])
        #     aux_err_phi = np.linalg.norm(phi_mod[y] - phi_aux[x])
        #     # aux_erk2 = np.sqrt(aux_erk2)
        #     erk.append(aux_erk2)
        #     err_phi.append(aux_err_phi)
        # erk = np.array(erk)
        # err_phi = np.array(err_phi)
        # s_erk.append(np.std(erk))
        # # err_phi = np.linalg.norm(phi_mod[y] - phi_project)
        # min_erk = np.amin(erk,axis=0)

        #        #TESTE
        #         teste_gamma = 0
        #         for x in range(0,K):
        #             teste_gamma = np.dot(u_cov[x],omega_project[x]) + teste_gamma
        #         teste_gamma = np.array(teste_gamma)
        #         # print("SHAPE DO TESTE GAMA "+str(teste_gamma.shape))
        #         teste_gamma = teste_gamma + psi
        #         # teste(teste_gamma.reshape(TAM_IMG))
        #         epsilon = np.sqrt(np.linalg.norm(phi_mod[y] - teste_gamma))
        #         print("VALOR DO EPSILON: "+str(epsilon))

        #         #Cálculo do Threshold
        #         theta_aux_i = []
        #         # theta_aux_j = []
        #         for i in range(0,phi_aux.shape[0]):
        #             for j in range(0,phi_aux.shape[0]):
        #                 theta_aux_i.append(np.linalg.norm(omega[i] - omega[j]))
        #         theta_aux_i = np.array(theta_aux_i)
        #         max_theta = np.amax(theta_aux_i,axis=0)
        #         max_theta = max_theta / 2
        #         # max_theta = max_theta * 0.3
        #         print("VALOR MÁXIMO DE THETA: "+str(max_theta))                
        #         #Cálculo do Threshold

        #         #TESTE PARA VER SE O ERRO MINIMO DE PHI AJUDA
        #         sigma1 = 0
        #         sigma2 = 0
        #         op_thresh = 0
        #         for z in range(0,phi_aux.shape[0]):
        #             if erk[z] == min_erk:
        #                 sigma1 = z
        #             if err_phi[z] == np.amin(err_phi,axis=0):
        #                 sigma2 = z
                
        #         if sigma1 == sigma2:
        #             op_thresh = sigma1
        #             print("NESSA SITUAÇÃO AMBOS SÃO IGUAIS E REINAM JUNTOS")
        #         else:
        #             op_thresh = sigma2
        #             print("NESSA SITUAÇÃO SIGMA 2 REINA SOZINHO, POIS SEMPRE ESTA CERTO")

        #         for z in range(0,phi_aux.shape[0]):
        #             if op_thresh == z:
        #                 # teste2((phi_mod[y] + psi).reshape(TAM_IMG),(phi_aux[z] + psi).reshape(TAM_IMG))
        #                 # teste3((phi_mod[y] + psi).reshape(TAM_IMG),(phi_aux[z] + psi).reshape(TAM_IMG),(phi_aux[sigma1] + psi).reshape(TAM_IMG))
        #                 print("O OMEGA É: "+str(z))
        #                 print("O VALOR É: "+str(erk[z]))
        #                 print("O ERRO MINIMO É: "+str(min_erk))
        #                 # print("VALOR DE OMEGA[z]: "+str(omega[z]))
        #                 print("VALOR NORMALIZADO: "+str(np.linalg.norm(omega[z])))
        #                 # print("VALOR DO OMEGA_PROJECT: "+str(omega_project))
        #                 print("VALOR NORMALIZADO: "+str(np.linalg.norm(omega_project)))
        #                 print("VALOR DO ERRO PHI: "+str(err_phi[z]))
        #                 print("VALOR MINIMO DO ERRO PHI: "+str(np.amin(err_phi,axis=0)))
        #                 if np.amin(err_phi,axis=0) >= E_PHI:
        #                     op = 0
        #                 else:
        #                     op = 1
        #                 # op2 = input("Certo - 1  Errado - 0 : ")
        #                 # op2 = 0
        #                 # flag_rec.append({"MIN_ERROR":str(min_erk),"NORM_OMEGA":str(np.linalg.norm(omega[z])),"NORM_PROJECT":str(np.linalg.norm(omega_project)),"PROJECT_OMEGA":str((np.linalg.norm(omega_project))-(np.linalg.norm(omega[z]))),"NORM_ERRO_PHI":str(np.linalg.norm(err_phi)),"MEDIA_ERRO_PHI":str(np.mean(err_phi)),"MIN_ERRO_PHI":str(np.amin(err_phi,axis=0)),"REC_SOLO":str(op),"REC":str(op2)})
                        
        #         #TESTE PARA VER SE O ERRO MINIMO DE PHI AJUDA



        #         # for z in range(0,phi_aux.shape[0]):
                    
        #         #     # if flag_rec != 0:    
        #         #     # print()
        #         #     if erk[z] == min_erk:
        #         #         # if (min_erk < max_theta) and (epsilon < max_theta) :
        #         #         #     print("A IMAGEM ESTA NOS PADROES")
        #         #         #     teste2((phi_mod[y] + psi).reshape(TAM_IMG),(phi_aux[z] + psi).reshape(TAM_IMG))
        #         #         print("O OMEGA É: "+str(z))
        #         #         print("O VALOR É: "+str(erk[z]))
        #         #         print("O ERRO MINIMO É: "+str(min_erk))
        #         #         print("VALOR DE OMEGA[z]: "+str(omega[z]))
        #         #         print("VALOR NORMALIZADO: "+str(np.linalg.norm(omega[z])))
        #         #         print("VALOR DO OMEGA_PROJECT: "+str(omega_project))
        #         #         print("VALOR NORMALIZADO: "+str(np.linalg.norm(omega_project)))
        #         #         print("SÓ PRA DESCARGO: "+str(np.linalg.norm(omega_project - omega[z])))
        #         #         # teste((phi_mod[y] + psi).reshape(TAM_IMG))
        #         #         # conte = gd.get_json2img(GDA,omega[z])
        #         #         # print(conte)
        #         #         print("Valor do Y: "+str(y)+" Valor do Z: "+str(z))
        #         #         teste2((phi_mod[y] + psi).reshape(TAM_IMG),(phi_aux[z] + psi).reshape(TAM_IMG))
        #         #         op = input("Certo - 1  Errado - 0 : ")
        #         #         flag_rec.append({"MIN_ERROR":str(min_erk),"NORM_OMEGA":str(np.linalg.norm(omega[z])),"NORM_PROJECT":str(np.linalg.norm(omega_project)),"PROJECT_OMEGA":str((np.linalg.norm(omega_project))-(np.linalg.norm(omega[z]))),"NORM_ERRO_PHI":str(np.linalg.norm(err_phi)),"MEDIA_ERRO_PHI":str(np.mean(err_phi)),"MIN_ERRO_PHI":str(np.amin(err_phi,axis=0)),"REC":str(op)})
        #         #         # flag_rec.append(input("0(errado) OU 1(certo)"))
        #         #         # teste3((phi_mod[y] + psi).reshape(TAM_IMG),(conte["PHI"]+psi).reshape(TAM_IMG),(phi_aux[z] + psi).reshape(TAM_IMG))
        #         #         # break
        #         # #TESTE

        # # w_aux = 0
        # # omega_project = []
        # # for x in range(0,K):
        # #     w = np.dot(u_cov[x],phi_mod)
        # #     w_aux = (w*u_cov[x]) + w_aux
        # #     omega_project.append(w)
        # # omega_project = np.array(omega_project)
        # # phi_project = w_aux
        # # erk = []
        # # for x in range(0,K):
        # #     aux_erk2 = np.linalg.norm(omega_project - omega[x])
        # #     erk.append(aux_erk2)
        # # erk = np.array(erk)
        # # err_phi = np.linalg.norm(phi_mod - phi_project)
        # # min_erk = np.amin(erk,axis=0)

    def img2json(self,name_arq,img_json,GDA,client,obj_name,tipo):
        # Seria certo salvar para cada pessoa o seu phi e omega, pois assim teria as métricas gerada de cada pessoa quando forem cadastradas no sistema,
        # pensando nisso, esse método salva o arquivo em um json e envia pro drive.
        # A função deve receber Phi_i , Omega_i e i.
        # A função deve receber o connect do Database.py e receber o id do arquivo do Google_Drive.py
        # A função deve enviar o nome e o objeto do arquivo para o Google_drive.py via parâmetro
        # As conexões das funções do Google_Drive.py e Database.py não podem ficar dentro da função, deve ser instanciada fora, devido a quantidade
        # repetida que as mesmas serão abertas.
        
        # client = db.conecta(self)
        # GDA = gd.connect()
        # img_json <-- Phi_i,Omega_i
        # obj_json <-- nome,id_drive 

        id_img = gd.create_img2json(name_arq,img_json,GDA)
        obj_json = {"Nome":obj_name,"Id_drive":id_img}
        db.gamma2db(self,client,obj_json,tipo)




    def img2db(self):
        client = db.conecta(self)
        images = ai.openImg2DB(self,DIRA)
        for i in range(0,len(images)):
            images[i] = cv.cvtColor(images[i],cv.COLOR_BGR2GRAY)
        [gamma, qnt_gamma] = ai.createDataMatrix(self,images)
        print(len(images))
        y = 1
        for x in range(0,qnt_gamma):
            img_list = gamma[x].tolist()
            if (x % 2 == 0):
                nome = "pessoa"+str(y)
                y += 1
            img_json = {"nome":nome,"pixel":img_list}

            # TESTE PARA SALVAR ARQUIVO E VER TAMANHO DELE
            # with open("teste2.json","w") as write_file:
                # json.dump(img_json,write_file,indent=4)
            # write_file.close()

            # img_json = {"nome":"foto"+str(x+1),"pixel":Binary(gamma[x])}
            

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
          
    def get_eigenfaces(self):
        #Geração do vetor Gamma(N^2 X 1)
        images = ai.openImg(self,DIRA)
        for i in range(0,len(images)):
            images[i] = cv.cvtColor(images[i],cv.COLOR_BGR2GRAY)
        [gamma, qnt_gamma] = ai.createDataMatrix(self,images)
        #Geração do vetor Gamma(N^2 X 1)

        #Não faz parte da obtenção das Eigenfaces, somente utilizada para a documentação.
        #Visualização das imagens que estão sendo utilizadas no sistema
        # for x in range(0,qnt_gamma):
        #     print("VALOR DE X: "+str(x))
        #     teste(gamma[x].reshape(TAM_IMG))
        #Visualização das imagens que estão sendo utilizadas no sistema
        
        #Limpeza do vetor images e coletor do Python(medida necessária para não estouro da memória do Python)
        images.clear()
        gc.collect()
        #Limpeza do vetor images e coletor do Python(medida necessária para não estouro da memória do Python)

        #Geração da face média Psi(N^2 X 1)
        psi = 0
        aux_psi = 0
        for x in range(0,qnt_gamma):
            aux_psi = gamma[x] + aux_psi
        psi = aux_psi/qnt_gamma
        # eigenfaces.save_psi(self,psi) 
        #Geração da face média Psi(N^2 X 1)

        #Geração do vetor Phi(Gamma - Psi)
        phi = []
        for y in range(0,qnt_gamma):
            phi.append(gamma[y] - psi)
        phi_aux = np.array(phi)
        #Geração do vetor Phi(Gamma - Psi)

        #Geração da matriz de covariância e vetor A(Phi)
        A = np.array(phi)
        phi.clear()
        A_T = A.transpose()
        Cov = np.dot(A,A_T)
        #Geração da matriz de covariância e vetor A(Phi)

        #Geração dos Eigenvectors e Eigenvalues de A_T A
        [eg_vle,eg_vec] = np.linalg.eig(Cov)
        #Geração dos Eigenvectors e Eigenvalues de A_T A

        #Organização do Eigenvector para obter os melhores vetores apartir dos maiores Eigenvalues
        args = np.argsort(eg_vle)
        args = args[::-1]
        args_vec = np.array(eg_vec[args])
        eg_vec = args_vec
        #Organização do Eigenvector para obter os melhores vetores apartir dos maiores Eigenvalues

        #Geração dos Eigenvectors de A A_T
        u_cov = []
        for x in range(0,eg_vec.shape[0]):
            u_cov.append(np.dot(A_T,eg_vec[x]))
        u_cov = np.array(u_cov)
        # eigenfaces.save_cov(self,u_cov)
        #Geração dos Eigenvectors de A A_T

        #Geração dos pesos de treino Omega
        aux_omegax = []
        for x in range (0,phi_aux.shape[0]):
            aux_omegay = []
            for y in range(0,K):
                aux_omegay.append(np.dot(u_cov[y],phi_aux[x]))
            aux_omegax.append(aux_omegay)
        omega = np.array(aux_omegax)
        aux_omegax.clear()
        aux_omegay.clear()
        #Geração dos pesos de treino Omega

        # Comentado pois não é necessário salvar no banco no momento
        # Fazer uma função a parte depois
        # Teste com a função de salvar no banco e drive
        # client = db.conecta(self)
        # GDA = gd.connect()
        # y = 1
        # for x in range(0,phi_aux.shape[0]):
        #     phi_list = phi_aux[x].tolist()
        #     omega_list = omega[x].tolist()
        #     if (x % 2 == 0):
        #         nome_arq = "pessoa"+str(y)
        #         y += 1
        #     img_json = '{"PHI":'+str(phi_list)+',"OMEGA":'+str(omega_list)+'}'
        #     eigenfaces.img2json(self,nome_arq,img_json,GDA,client,nome_arq,"PHI")

        #Não faz parte da obtenção das Eigenfaces, somente utilizada para a documentação.

        #Geração da tabela de Omegas em json
        # omega_save = "omega"
        # omega_json = []
        # with open("omega"+str(K)+".json","w") as json_file:
            # for x in range(0,K):
                # omega_json.append({omega_save + str(x+1):omega[x].tolist()}) 
            # json.dump(omega_json,json_file,indent=4)
        #Geração da tabela de Omegas em json

        #Geração da tabela de erros euclidianos

        #Teste com faces modificadas, porém todas do treinamento
        images = ai.openImg(self,DIRA_MOD)
        #Teste com faces modificadas, porém todas do treinamento
        
        #Teste com faces exatamente do treinamento
        # images = ai.openImg(self,DIRA_TESTE)
        #Teste com faces exatamente do treinamento

        #Teste com faces desconhecidas pelo sistema
        # images = ai.openImg(self,DIRA_TESTE2)
        #Teste com faces desconhecidas pelo sistema

        for i in range(0,len(images)):
            images[i] = cv.cvtColor(images[i],cv.COLOR_BGR2GRAY)
        [gamma_mod, qnt_gamma_mod] = ai.createDataMatrix(self,images)
        images.clear()

        phi_mod = []
        for x in range(0,qnt_gamma_mod):
            phi_mod.append(gamma_mod[x] - psi)
        phi_mod = np.array(phi_mod)
        
        omega_json = []
        flag_rec = []
        s_project = []
        s_erk = []
        diff = []
        # GDA = gd.connect()
        with open(PATH_METRI+"erros"+str(K)+".json","w") as json_file:
            for y in range(0,qnt_gamma_mod):
                w_aux = 0
                omega_project = []
                for x in range(0,K):
                    w = np.dot(u_cov[x],phi_mod[y])
                    w_aux = (w*u_cov[x]) + w_aux
                    omega_project.append(w)
                omega_project = np.array(omega_project)
                s_project.append(np.std(omega_project))
                phi_project = w_aux
                erk = []
                err_phi = []
                for x in range(0,phi_aux.shape[0]):
                    aux_erk2 = np.linalg.norm(omega_project - omega[x])
                    aux_err_phi = np.linalg.norm(phi_mod[y] - phi_aux[x])
                    # aux_erk2 = np.sqrt(aux_erk2)
                    erk.append(aux_erk2)
                    err_phi.append(aux_err_phi)
                erk = np.array(erk)
                err_phi = np.array(err_phi)
                s_erk.append(np.std(erk))
                # err_phi = np.linalg.norm(phi_mod[y] - phi_project)
                min_erk = np.amin(erk,axis=0)

               #TESTE
                teste_gamma = 0
                for x in range(0,K):
                    teste_gamma = np.dot(u_cov[x],omega_project[x]) + teste_gamma
                teste_gamma = np.array(teste_gamma)
                # print("SHAPE DO TESTE GAMA "+str(teste_gamma.shape))
                teste_gamma = teste_gamma + psi
                # teste(teste_gamma.reshape(TAM_IMG))
                epsilon = np.sqrt(np.linalg.norm(phi_mod[y] - teste_gamma))
                print("VALOR DO EPSILON: "+str(epsilon))

                #Cálculo do Threshold
                theta_aux_i = []
                # theta_aux_j = []
                for i in range(0,phi_aux.shape[0]):
                    for j in range(0,phi_aux.shape[0]):
                        theta_aux_i.append(np.linalg.norm(omega[i] - omega[j]))
                theta_aux_i = np.array(theta_aux_i)
                max_theta = np.amax(theta_aux_i,axis=0)
                max_theta = max_theta / 2
                # max_theta = max_theta * 0.3
                print("VALOR MÁXIMO DE THETA: "+str(max_theta))                
                #Cálculo do Threshold

                #TESTE PARA VER SE O ERRO MINIMO DE PHI AJUDA
                sigma1 = 0
                sigma2 = 0
                op_thresh = 0
                for z in range(0,phi_aux.shape[0]):
                    if erk[z] == min_erk:
                        sigma1 = z
                    if err_phi[z] == np.amin(err_phi,axis=0):
                        sigma2 = z
                
                if sigma1 == sigma2:
                    op_thresh = sigma1
                    print("NESSA SITUAÇÃO AMBOS SÃO IGUAIS E REINAM JUNTOS")
                else:
                    op_thresh = sigma2
                    print("NESSA SITUAÇÃO SIGMA 2 REINA SOZINHO, POIS SEMPRE ESTA CERTO")

                #PARTE MODIFICADA PARA PODER ANALISAR OS PESOS DOS ERROS DE CADA FACE INDIVIDUALMENTE
                
                for z in range(0,phi_aux.shape[0]):
                    if op_thresh == z:
                        # teste2((phi_mod[y] + psi).reshape(TAM_IMG),(phi_aux[z] + psi).reshape(TAM_IMG))
                        # teste3((phi_mod[y] + psi).reshape(TAM_IMG),(phi_aux[z] + psi).reshape(TAM_IMG),(phi_aux[sigma1] + psi).reshape(TAM_IMG))
                        print("O OMEGA É: "+str(z))
                        print("O VALOR É: "+str(erk[z]))
                        print("O ERRO MINIMO É: "+str(min_erk))
                        # print("VALOR DE OMEGA[z]: "+str(omega[z]))
                        print("VALOR NORMALIZADO: "+str(np.linalg.norm(omega[z])))
                        # print("VALOR DO OMEGA_PROJECT: "+str(omega_project))
                        print("VALOR NORMALIZADO: "+str(np.linalg.norm(omega_project)))
                        print("VALOR DO ERRO PHI: "+str(err_phi[z]))
                        print("VALOR MINIMO DO ERRO PHI: "+str(np.amin(err_phi,axis=0)))
                        if np.amin(err_phi,axis=0) >= E_PHI:
                            op = 0
                        else:
                            op = 1
                        # op2 = input("Certo - 1  Errado - 0 : ")
                            # op2 = 0
                        if op == 0:
                            with open(PATH_METRI + "diff_erro"+str(z)+".json","w") as save_diff:            
                                diff.append({"INDICE":str(z),"OMEGA_PROJECT":omega_project.tolist(),"OMEGA":omega[z].tolist(),"DIFFA_OMEGA":(omega_project-omega[z]).tolist(),"DIFFB_OMEGA":(omega_project-omega[sigma1]).tolist(),"PHI_AUX":phi_aux[z].tolist(),"PHI_MOD":phi_mod.tolist(),"DIFFA_PHI":(phi_mod-phi_aux[z]).tolist(),"DIFFB_PHI":(phi_mod-phi_aux[z]).tolist()})
                                json.dump(diff,save_diff,indent=4)
                                diff.clear()
                    
                #PARTE MODIFICADA PARA PODER ANALISAR OS PESOS DOS ERROS DE CADA FACE INDIVIDUALMENTE


                # for z in range(0,phi_aux.shape[0]):
                #     if op_thresh == z:
                #         # teste2((phi_mod[y] + psi).reshape(TAM_IMG),(phi_aux[z] + psi).reshape(TAM_IMG))
                #         teste3((phi_mod[y] + psi).reshape(TAM_IMG),(phi_aux[z] + psi).reshape(TAM_IMG),(phi_aux[sigma1] + psi).reshape(TAM_IMG))
                #         print("O OMEGA É: "+str(z))
                #         print("O VALOR É: "+str(erk[z]))
                #         print("O ERRO MINIMO É: "+str(min_erk))
                #         # print("VALOR DE OMEGA[z]: "+str(omega[z]))
                #         print("VALOR NORMALIZADO: "+str(np.linalg.norm(omega[z])))
                #         # print("VALOR DO OMEGA_PROJECT: "+str(omega_project))
                #         print("VALOR NORMALIZADO: "+str(np.linalg.norm(omega_project)))
                #         print("VALOR DO ERRO PHI: "+str(err_phi[z]))
                #         print("VALOR MINIMO DO ERRO PHI: "+str(np.amin(err_phi,axis=0)))
                #         if np.amin(err_phi,axis=0) >= E_PHI:
                #             op = 0
                #         else:
                #             op = 1
                #         op2 = input("Certo - 1  Errado - 0 : ")
                #         # op2 = 0
                #         flag_rec.append({"MIN_ERROR":str(min_erk),"NORM_OMEGA":str(np.linalg.norm(omega[z])),"NORM_PROJECT":str(np.linalg.norm(omega_project)),"PROJECT_OMEGA":str((np.linalg.norm(omega_project))-(np.linalg.norm(omega[z]))),"NORM_ERRO_PHI":str(np.linalg.norm(err_phi)),"MEDIA_ERRO_PHI":str(np.mean(err_phi)),"MIN_ERRO_PHI":str(np.amin(err_phi,axis=0)),"REC_SOLO":str(op),"REC":str(op2)})
                        
                #TESTE PARA VER SE O ERRO MINIMO DE PHI AJUDA



                # for z in range(0,phi_aux.shape[0]):
                    
                #     # if flag_rec != 0:    
                #     # print()
                #     if erk[z] == min_erk:
                #         # if (min_erk < max_theta) and (epsilon < max_theta) :
                #         #     print("A IMAGEM ESTA NOS PADROES")
                #         #     teste2((phi_mod[y] + psi).reshape(TAM_IMG),(phi_aux[z] + psi).reshape(TAM_IMG))
                #         print("O OMEGA É: "+str(z))
                #         print("O VALOR É: "+str(erk[z]))
                #         print("O ERRO MINIMO É: "+str(min_erk))
                #         print("VALOR DE OMEGA[z]: "+str(omega[z]))
                #         print("VALOR NORMALIZADO: "+str(np.linalg.norm(omega[z])))
                #         print("VALOR DO OMEGA_PROJECT: "+str(omega_project))
                #         print("VALOR NORMALIZADO: "+str(np.linalg.norm(omega_project)))
                #         print("SÓ PRA DESCARGO: "+str(np.linalg.norm(omega_project - omega[z])))
                #         # teste((phi_mod[y] + psi).reshape(TAM_IMG))
                #         # conte = gd.get_json2img(GDA,omega[z])
                #         # print(conte)
                #         print("Valor do Y: "+str(y)+" Valor do Z: "+str(z))
                #         teste2((phi_mod[y] + psi).reshape(TAM_IMG),(phi_aux[z] + psi).reshape(TAM_IMG))
                #         op = input("Certo - 1  Errado - 0 : ")
                #         flag_rec.append({"MIN_ERROR":str(min_erk),"NORM_OMEGA":str(np.linalg.norm(omega[z])),"NORM_PROJECT":str(np.linalg.norm(omega_project)),"PROJECT_OMEGA":str((np.linalg.norm(omega_project))-(np.linalg.norm(omega[z]))),"NORM_ERRO_PHI":str(np.linalg.norm(err_phi)),"MEDIA_ERRO_PHI":str(np.mean(err_phi)),"MIN_ERRO_PHI":str(np.amin(err_phi,axis=0)),"REC":str(op)})
                #         # flag_rec.append(input("0(errado) OU 1(certo)"))
                #         # teste3((phi_mod[y] + psi).reshape(TAM_IMG),(conte["PHI"]+psi).reshape(TAM_IMG),(phi_aux[z] + psi).reshape(TAM_IMG))
                #         # break
                # #TESTE

                omega_json.append({"ERRO_OMEGA":erk.tolist(),"MIN_ERRO_OMEGA":str(min_erk),"ERRO_PHI":err_phi.tolist(),"NORM_ERRO_PHI":str(np.linalg.norm(err_phi)),"MEDIA_ERRO_PHI":str(np.mean(err_phi)),"MIN_ERRO_PHI":str(np.amin(err_phi,axis=0)),"Media":str(np.mean(erk)),"Desvio":str(np.std(erk))}) 
            json.dump(omega_json,json_file,indent=4)
        
        # with open(PATH_METRI + "norms"+str(K)+".json","w") as norm_json:
            # json.dump(flag_rec,norm_json,indent=4)

        #Geração da tabela de erros euclidianos

        #GERAÇÃO DO GRAFICO DE ACERTOS NOVOS
        with open(PATH_METRI + "norms70.json","r") as e_phi:
            val = json.loads(e_phi.read())
            win = []
            win_val = []
            lose = []
            lose_val = []
            valores = []
            FN = 0
            FP = 0
            EQ = 0
            for ind in range(0,len(val)):
                # win_val.append(float(val[ind]["MIN_ERRO_PHI"]))
                if val[ind]["REC"] > val[ind]["REC_SOLO"]:
                    FN += 1
                elif val[ind]["REC"] < val[ind]["REC_SOLO"]:
                    FP += 1
                else:
                    EQ += 1

                if val[ind]["REC_SOLO"] == "1":
                    # win.append("1")
                    # valores.append(ind)
                    win.append(float(val[ind]["MIN_ERRO_PHI"]))
                else:
                    # lose.append("0")
                    lose.append(float(val[ind]["MIN_ERRO_PHI"]))
                if val[ind]["REC"] == "1":
                    # win.append("1")
                    # valores.append(ind)
                    win_val.append(float(val[ind]["MIN_ERRO_PHI"]))
                else:
                    # lose.append("0")
                    lose_val.append(float(val[ind]["MIN_ERRO_PHI"]))
        e_phi.close()
        
        # print(win_val)
        # print(lose_val)
        plt.subplot(2,1,1)
        plt.title("ERRO PHI = "+str(E_PHI))
        plt.plot(win,win,"ob")
        plt.plot(lose_val,lose_val,"dy")
        plt.legend(['Acertos S/ Supervisão',"Erros C/ Supervisão"],loc='best')
        plt.text(40,35,"Quantidade de Acertos S/ Supervisão: "+str(len(win)))
        plt.text(40,30,"Quantidade de Erros C/ Supervisão: "+str(len(lose_val)))
        plt.subplot(2,1,2)
        plt.plot(win_val,win_val,"og")
        plt.plot(lose,lose,"dr")
        plt.legend(['Acertos C/ Supervisão',"Erros S/ Supervisão"],loc='best')
        plt.text(44,40,"Quantidade de Acertos C/ Supervisão: "+str(len(win_val)))
        plt.text(44,38,"Quantidade de Erros S/ Supervisão: "+str(len(lose)))
        plt.show()

        #GERAÇÃO DO GRAFICO DE ACERTOS NOVOS

        #Cálculo de Media,Mediana,Desvio Padrão, Variância
        s = []
        with open(PATH_METRI+"omega70.json","r") as calc:
            valores = json.loads(calc.read())
            for qnt in range(0,len(valores)):
                print("NUMERO DO OMEGA: "+str(qnt))
                print("MEDIA: "+str(np.mean(valores[qnt]["omega"+str(qnt+1)])))
                print("MEDIANA: "+str(np.median(valores[qnt]["omega"+str(qnt+1)])))
                print("MEDIA ARITIMÉTICA: "+str(np.average(valores[qnt]["omega"+str(qnt+1)])))
                print("DESVIO: "+str(np.std(valores[qnt]["omega"+str(qnt+1)])))
                s.append(np.std(valores[qnt]["omega"+str(qnt+1)]))
                print("VARIANCIA: "+str(np.var(valores[qnt]["omega"+str(qnt+1)])))
                # valores[qnt]["omega"+str(qnt+1)]
        calc.close()
        print("Desvio Padrao de s: "+str(np.std(s)))
        plt.title("DESVIO PADRÃO DOS OMEGAS DO TREINAMENTO")
        plt.xlabel("INDICE DE OMEGA")
        plt.ylabel("DESVIO PADRÃO S")
        plt.plot(s,"ob")
        plt.show()
        print("DESVIO PADRÂO DE CADA OMEGA PROJETADO: "+str(s_project))
        plt.title("DESVIO PADRÃO DOS OMEGAS PROJETADOS NO ESPAÇO DE FACES")
        plt.xlabel("INDICE DE OMEGA")
        plt.ylabel("DESVIO PADRÃO S")
        plt.plot(s_project,"ob")
        plt.show()
        print("DESVIO PADRÂO DE CADA ERRO OMEGA_PROJETADO - OMEGA: "+str(s_erk))
        plt.title("DESVIO PADRÃO DO ERRO")
        plt.xlabel("INDICE DE ERROS")
        plt.ylabel("DESVIO PADRÃO S")
        plt.plot(s_erk,"ob")
        plt.show()
        #Cálculo de Media,Mediana,Desvio Padrão, Variância

        #VALOR MÉDIO DO MIN_ERROR
        # media = []
        # with open("norms70.json","r") as med:
        #     med_read = json.loads(med.read())
        #     for b in range(0,len(med_read)):
        #         media.append(float(med_read[b]["MIN_ERROR"]))
        #     # media = np.array(media)
        #     # print(media)
        #     media_aux = np.mean(media)
        #     print("VALOR MEDIO DO ERRO MINIMO DE TODAS AS IMAGENS DE TESTE: "+str(media_aux))
        #VALOR MÉDIO DO MIN_ERROR

        #Geração de gráficos
        with open("med.json","r") as graph:
            graph_read = json.loads(graph.read())
            x_line = []
            y_line = []
            for a in range(0,len(graph_read)):
                x_line.append(graph_read[a]["K"])
                y_line.append(graph_read[a]["WIN"])
            # plt.subplot(2,1,1)
            plt.title("K x ACERTOS")
            plt.xlabel("VALOR DE K")
            plt.ylabel("VALOR DE ACERTOS")
            plt.plot(x_line,y_line)
            plt.show()
            x_line = []
            y_line = []
            for a in range(0,len(graph_read)):
                x_line.append(graph_read[a]["K"])
                y_line.append(graph_read[a]["LOSE"])
            # plt.subplot(2,1,2)
            # plt.clf()
            plt.title("K x ERROS")
            plt.xlabel("VALOR DE K")
            plt.ylabel("VALOR DE ERROS")
            plt.plot(x_line,y_line)
            plt.show()
        #Geração de gráficos

        print("VALOR DOS FALSOS POSITIVOS: "+str(FP)+" E NEGATIVOS: "+str(FN)+" QNT DE IGUAIS: "+str(EQ))

        # Geração de tabelas de diferença de erros
        # with open("med.txt","w") as write_txt:
        #     for a in range(61,80):
        #         with open("norms"+str(a)+".json") as read_json:
        #             norms = json.loads(read_json.read())
        #             val_med_0 = 0
        #             val_med_1 = 0
        #             for b in range(0,len(norms)):
        #                 if (norms[b]["REC"] == "0"):
        #                     print("SO PRA TESTAR QUE TA 0 - VALOR DO A: "+str(a))
        #                     val_med_0 += 1
        #                 else:
        #                     print("SO PRA TESTAR QUE TA 1 - VALOR DO A: "+str(a))
        #                     val_med_1 += 1
        #             write_txt.writelines("VALOR DE K = "+str(a)+" Quantidade de ACERTOS: "+str(val_med_1) + " Quantidade de ERROS: "+str(val_med_0)+"\n")
        #         read_json.close()
        # write_txt.close()      
        #Geração de tabelas de diferença de erros
        
        #Salvar as Eigenfaces
        # sv_name = "eigen"
        # for x in range(0,10):
            # norm = cv.normalize(u_cov[x].reshape(TAM_IMG), None, alpha=0, beta=1,norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)
            # norm = norm*255.0
            # cv.imwrite((sv_name+str(x+1)+".jpg"),norm) 
        #Salvar as Eigenfaces
        
        #Salvar Psi
        # norm = cv.normalize(psi.reshape(TAM_IMG), None, alpha=0, beta=1,norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)
        # norm = norm*255.0
        # cv.imwrite("psi.jpg",norm)
        #Salvar Psi
        
        #Salvar Phi
        # for x in range(0,10):
            # norm = cv.normalize(phi_aux[x].reshape(TAM_IMG), None, alpha=0, beta=1,norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)
            # norm = norm*255.0
            # cv.imwrite("phi"+str(x)+".jpg",phi_aux[x].reshape(TAM_IMG))
        #Salvar Phi
        #Não faz parte da obtenção das Eigenfaces, somente utilizada para a documentação.
        
        #Comentado, pois é necessário fazer uma função própria para reconhecimento, porém esta funcionando
        # images = ai.openImg(self,DIRA_MOD)
        # for i in range(0,len(images)):
        #     images[i] = cv.cvtColor(images[i],cv.COLOR_BGR2GRAY)
        # [gamma_mod, qnt_gamma_mod] = ai.createDataMatrix(self,images)
        # phi_mod = gamma_mod[0] - psi
        # w_aux = 0
        # omega_project = []
        # for x in range(0,K):
        #     w = np.dot(u_cov[x],phi_mod)
        #     w_aux = (w*u_cov[x]) + w_aux
        #     omega_project.append(w)
        # omega_project = np.array(omega_project)
        # phi_project = w_aux
        # erk = []
        # for x in range(0,K):
        #     aux_erk2 = np.linalg.norm(omega_project - omega[x])
        #     erk.append(aux_erk2)
        # erk = np.array(erk)
        # err_phi = np.linalg.norm(phi_mod - phi_project)
        # min_erk = np.amin(erk,axis=0)
        #Comentado, pois é necessário fazer uma função própria para reconhecimento, porém esta funcionando
    
    def eig_mod(self):
        images = ai.openImg(self,DIRA)
        for i in range(0,len(images)):
            images[i] = cv.cvtColor(images[i],cv.COLOR_BGR2GRAY)
        [gamma, qnt_gamma] = ai.createDataMatrix(self,images)
        
        images.clear()
        gc.collect()

        psi = 0
        aux_psi = 0
        for x in range(0,qnt_gamma):
            aux_psi = gamma[x] + aux_psi
        psi = aux_psi/qnt_gamma
        print("VALOR DO SHAPE DO PSI: "+str(psi.shape))
        phi = []
        for y in range(0,qnt_gamma):
            phi.append(gamma[y] - psi)
        phi_aux = np.array(phi)
        print("PHI_AUX SHAPE "+ str(phi_aux.shape))
        A = np.array(phi)
        phi.clear()
        print(A.shape)
        A_T = A.transpose()
        print(A_T.shape)
        Cov = np.dot(A,A_T)
        print(Cov.shape)
        [eg_vle,eg_vec] = np.linalg.eig(Cov)
        #EG_VEC é o vi do STEP 6.2

        #MODIFICAÇÃO FEITA QUE PARECE QUE MELHOROU UM POUCO            
        args = np.argsort(eg_vle)
        # print("ANTES DO REVERSE: "+str(args))
        args = args[::-1]
        # print("DEPOIS DO REVERSE: "+str(args))
        print("ANTES VALOR DE EG_VEC" + str(eg_vec) + "SHAPE DO EG_VEC " + str(eg_vec.shape))
        args_vec = np.array(eg_vec[args])
        eg_vec = args_vec
        #MODIFICAÇÃO FEITA QUE PARECE QUE MELHOROU UM POUCO

        print("VALOR DE EG_VEC" + str(eg_vec) + "SHAPE DO EG_VEC " + str(eg_vec.shape))
        print("VALOR DE EG_VLE" + str(eg_vle) + "SHAPE DO EG_VLE " + str(eg_vle.shape))
        u_bostao = []
        for x in range(0,eg_vec.shape[0]):
            u_bostao.append(np.dot(A_T,eg_vec[x]))

        u_bostao = np.array(u_bostao)
        print("SHAPE DO U_BOSTAO: "+str(u_bostao.shape))
        

        aux_omegax = []
        for x in range (0,phi_aux.shape[0]):
            aux_omegay = []
            for y in range(0,K):
                aux_omegay.append(np.dot(u_bostao[y],phi_aux[x]))
            aux_omegax.append(aux_omegay)
        omega = np.array(aux_omegax)
        aux_omegax.clear()
        aux_omegay.clear()
        # print("OMEGA: " +str(omega))
        print("OMEGA SHAPE: " +str(omega.shape))


        # for x in range(0,K):
        #     aux_omegax = np.linalg.norm(omega[x])
        #     aux_omegay = omega[x]/aux_omegax
        #     omega[x] = aux_omegay

        # print("OMEGA NOMR: " +str(omega))
        # omega = cv.normalize(omega, None, alpha=-1, beta=1,norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)
        print("OMEGA: " +str(omega))

        #TESTE DE RECONHECIMENTO
        images = ai.openImg(self,DIRA_MOD)
        for i in range(0,len(images)):
            images[i] = cv.cvtColor(images[i],cv.COLOR_BGR2GRAY)
        [gamma_mod, qnt_gamma_mod] = ai.createDataMatrix(self,images)
        phi_mod = gamma_mod[0] - psi
        w_aux = 0
        omega_project = []
        for x in range(0,K):
            w = np.dot(u_bostao[x],phi_mod)
            w_aux = (w*u_bostao[x]) + w_aux
            # teste(w_aux.reshape(TAM_IMG))
            omega_project.append(w)
        omega_project = np.array(omega_project)
        phi_project = w_aux
        erk = []
        for x in range(0,K):
            aux_erk2 = np.linalg.norm(omega_project - omega[x])
            erk.append(aux_erk2)
        erk = np.array(erk)
        err_phi = np.linalg.norm(phi_mod - phi_project)
        print("Valor do erro: "+str(erk))
        min_erk = np.amin(erk,axis=0)
        print("MENOR ERRO DE OMEGA: "+str(min_erk)+" ERRO DE PHI: "+str(err_phi))
        
        # phi_project_n = np.linalg.norm(phi_project)
        # phi_project_n = phi_project/phi_project_n
        # phi_project = (phi_project*255.0)
        # aux_psiquico = np.linalg.norm(psi)
        # aux_psiquico = psi/aux_psiquico
        # aux_psiquico = aux_psiquico*255.0
        # teste((phi_project+aux_psiquico).reshape(TAM_IMG))
        
        
        #TESTE DE RECONHECIMENTO

        for x in range(0,omega.shape[0]):
            aux_tst = 0
            for y in range(0,K):
                # teste2(u_bostao[y].reshape(TAM_IMG),np.dot(omega[x][y],u_bostao[y]).reshape(TAM_IMG))
                aux_tst = (np.dot(omega[x][y],u_bostao[y])) + aux_tst
                
                # aux_bosta = np.linalg.norm(aux_tst)
                # aux_bosta = aux_tst/aux_bosta
                # aux_tst = aux_bosta*255.0
                
                # teste(aux_tst.reshape(TAM_IMG))

            aux_bosta = np.linalg.norm(aux_tst)
            aux_bosta = aux_tst/aux_bosta

            aux_psiquico = np.linalg.norm(psi)
            aux_psiquico = psi/aux_psiquico

            aux_psiquico = aux_psiquico*255.0
            aux_tst = aux_bosta*255.0

            aux_tst = aux_tst + aux_psiquico
            teste3((phi_aux[x]+psi).reshape(TAM_IMG),u_bostao[x].reshape(TAM_IMG),aux_tst.reshape(TAM_IMG))
    
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
        print("VALOR DO SHAPE DO PSI: "+str(psi.shape))
        phi = []
        for y in range(0,qnt_gamma):
            phi.append(gamma[y] - psi)
        # matrix_cov = 0
        # aux_cov = []
        phi_aux = np.array(phi)
        # phi_aux = cv.normalize(phi_aux, None, alpha=0, beta=1,norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)
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
        phi.clear()
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

        #O INICIO DESSA MERDA QUE ME CANSOU
        u_bostao = []
        for x in range(0,eg_vec.shape[0]):
            u_bostao.append(np.dot(A_T,eg_vec[x]))

        u_bostao = np.array(u_bostao)
        print("SHAPE DO U_BOSTAO: "+str(u_bostao.shape))
        # print("SÒ PRA VER: "+str(np.linalg.norm(u_bostao[0]/np.linalg.norm(u_bostao[0]))))

        #A sequência de código serve para salvar as eigenfaces, não precisa ser executada a todo o momento,
        # necessária somente para a manter guardado no banco e na documentação. Sendo assim pode ser comentada do código.
        # for x in range(0,K):
            # normimg = cv.normalize(u_bostao[x], None, alpha=0, beta=1,norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)
            # normimg = normimg/255.0
            # teste2(phi_aux[x].reshape(TAM_IMG),u_t[x].reshape(TAM_IMG))
            # teste(normimg.reshape(TAM_IMG))
            # name = "eigen"+str(x)+".jpg"
            # ai.saveImg(self,u_bostao[x].reshape(TAM_IMG),name,PATH_SAVE)

        # for x in range(0,u_bostao.shape[0]):
        #     aux_bosta = np.linalg.norm(u_bostao[x])
        #     aux_bosta = u_bostao[x]/aux_bosta
        #     u_bostao[x] = aux_bosta*255.0

        #O FIM DESSA MERDA QUE ME CANSOU



        u = np.dot(A_T,eg_vec)
        print("SHAPE DA BOSTA DO U:"+str(u.shape))
        # aux = 0
        # for x in range(0,eg_vec.shape[0]):
            # aux = (np.dot(eg_vec[x].reshape((196,1)),phi_aux[x].reshape((1,90000)))) + aux

        # u = aux
        # u = cv.normalize(u, None, alpha=0, beta=1,norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)
        # auxiliar
        uteste = u
        for x in range(0,u.shape[0]):
            u_n = np.linalg.norm(uteste[x])
            u_n = uteste[x]/u_n
            if np.linalg.norm(u_n) == 1:
                # print(str(x)+" TA NORMALIZADO OLHA -->"+str(np.linalg.norm(u_n)))
                uteste[x] = u_n
                # print(u[x])
                # break
            else:
                # print(str(x)+" NÃO TA NORMALIZADO")
                uteste[x] = 0
            # print("NORMA DO u["+str(x)+"]:"+str(u_n)+"unitario de u: "+str(u[x]/u_n))
        # uteste = np.sort(uteste)
        print(uteste)
        aux_M = []
        for x in range(0,u.shape[0]):
            if np.linalg.norm(uteste[x]) == 1:
                aux_M.append(uteste[x])
        aux_M = np.array(aux_M)
        print("SHAPE DO NOVO BOSTA: "+str(aux_M.shape))
        print("SHAPE DO NOVO BOSTA T: "+str(aux_M.T.shape))
        # for x in range(u.shape[0]-1,0,-1):
            # print("SÓ PRA TESTE DO MAIOR: "+str(uteste[u.shape[0]-x])+" DO MENOR: "+str(uteste[x]))
            # uteste[u.shape[0]-x],uteste[x] = uteste[x],uteste[u.shape[0]-x]
        # print("SHAPE DESSA CARALHA: "+str(uteste[x].shape))
        # print("SHAPE DESSA CARALHA T: "+str(uteste))
        uteste = uteste.T
        # aux_M = aux_M.T
            

        print("U shape: "+str(u.shape))
        print("PHI_AUX SHAPE: "+str(phi_aux.shape))
        u_t = u.transpose()

        print("u_t | QUAL O TIPO DIFERENTE DE BOSTA É ESSA:"+str(u_t.dtype))
        print("uteste | QUAL O TIPO DIFERENTE DE BOSTA É ESSA: "+str(uteste.dtype))
        # # uaux_t = uaux.T
        # print("U_T shape: "+str(u_t.shape))
        # print("VALOR DE U_T[0]: "+str(u_t[0])+ " MAIS A NORMALIZAÇÂO DO VALOR "+str(np.linalg.norm(u_t[0])))
        # w = np.dot(u_t,np.transpose(phi_aux))
        # print("w SHAPE: "+str(w.shape))
        # print("SHAPE DO w[0]: "+str(w[0].shape) + " | SHAPE DO u_t[0]: "+str(u[0].shape))
        # u_mod = []
        # for x in range(0,u_t.shape[0]):
        #     u_mod.append(np.linalg.norm(u_t[x]))
        # print(u_mod)
        # teste(phi_aux[47].reshape(TAM_IMG))

        #INICIO TESTE DO EIGENVALUE PRA ACHAR OS MAIORES EIGENVECTORS#
        # aux_zero = np.zeros((196,90000))
        # sorte = np.argsort(eg_vle)
        # aux_sorte1 = u_t[sorte]
        # aux_sorte2 = eg_vec[sorte]
        # print("VALOR DO SORTE: "+str(sorte))
        # print("VALOR DO AUX_SORTE1: "+str(aux_sorte1.shape))
        # print("VALOR DO u_t: "+str(u_t))
        # print("VALOR DO AUX_SORTE2: "+str(aux_sorte2))
        # # z = 0
        # for x in range(195,0,-1):
        #     aux_zero[195-x] = aux_sorte1[x]
            # z += 1

        # ord = False
        # while not ord:
        #     ord = True
        #     for x in range(0,(eg_vle.shape[0])-1):
        #         if (eg_vle[x] < eg_vle[x+1]):
        #             u_t[x],u_t[x+1] = u_t[x+1],u_t[x]
        #             print("VALOR DO X: "+str(x))
        #             ord = False
        #FIM TESTE DO EIGENVALUE PRA ACHAR OS MAIORES EIGENVECTORS#


        # ord = False
        # while not ord:
        #     ord = True
        #     for x in range(0,len(u_mod)-1):
        #         if (u_mod[x] < u_mod[x+1]):
        #             u_mod[x],u_mod[x+1] = u_mod[x+1],u_mod[x]
        #             u_t[x],u_t[x+1] = u_t[x+1],u_t[x]
        #             phi_aux[x],phi_aux[x+1]=phi_aux[x+1],phi_aux[x]
        #             ord = False
        # print("MODIFICADO: "+str(u_mod))
        # teste(phi_aux[47].reshape(TAM_IMG))

        #INICIO TESTE DE REPRESENTAÇÃO DAS EIGENFACES NO FACESPACE
        
        print("SHAPE DO phi_aux: "+str(phi_aux.shape))
        # print("SHAPE DO u_tT[0]: "+str(np.transpose(u_t[0]).shape))
        print("VALOR DO phi_aux[0]: "+str(phi_aux[0]))
        # teste(phi_aux[0].reshape(TAM_IMG))

        aux_omegax = []

        # for x in range(0,phi_aux.shape[0]):
        #     norm_phi = np.linalg.norm(phi_aux[x])
        #     phi_aux[x] = phi_aux[x]/norm_phi


        for x in range (0,phi_aux.shape[0]):
            aux_omegay = []
            for y in range(0,K):
                # U_NORM = cv.normalize(u_t[y], None, alpha=0, beta=1,norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)
                # U_NORM = cv.normalize(aux_zero[y], None, alpha=0, beta=1,norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)
                # print("QUE BOSTE DE SHAPE TEM ESSE CARALHO QUE O OUTRO NÂO TEM:"+str(u_t[y].shape)+" E ESSA OUTRA BOSTA AQUI: "+str(phi_aux[x].shape))
                # aux_omegay.append(np.dot(u_t[y],phi_aux[x]))
                # aux_omegay.append(np.dot(uteste[y],phi_aux[x]))
                # aux_omegay.append(np.dot(aux_M[y],phi_aux[x]))
                # teste(u_bostao[y].reshape(TAM_IMG))
                aux_omegay.append(np.dot(u_bostao[y],phi_aux[x]))
                # print("OMEGA["+str(x)+"]["+str(y)+"]: "+str(np.dot(u_bostao[y],phi_aux[x])))
                
                # aux_omegay.append(np.dot(aux_zero[y],phi_aux[x]))
                # bosta = np.array([uaux[y]])
                # bosta2 = np.array([phi_aux[x]])
                # print("BOSTA"+str(bosta.shape))
                # print("BOSTA2"+str(bosta2.shape))
                
                # aux_omegay.append(np.dot(aux_zero[y],cv.normalize(phi_aux[x], None, alpha=0, beta=1,norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)))
                # aux_omegay.append(np.dot(np.transpose(U_NORM),phi_aux[x]))
            # print("OMEGA Y: "+str(aux_omegay))
            aux_omegax.append(aux_omegay)
            # print("VALOR DO OMEGA X: "+str(aux_omegax))
            # aux_omegay.clear()
        omega = np.array(aux_omegax)
        aux_omegax.clear()
        aux_omegay.clear()
        # omega = aux_omegax
        # aux_omegax.clear()
        print("OMEGA: " +str(omega))
        print("OMEGA SHAPE: " +str(omega.shape))
        
        aux_tst = 0
        
        print("VALOR DO u_t[0]: "+str(u_t[0])+" E SEU NORM: "+str(np.linalg.norm(u_t[0])))
        # teste2(aux_zero[0].reshape(TAM_IMG),cv.normalize(aux_zero[0], None, alpha=0, beta=1,norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F).reshape(TAM_IMG))
        # teste2(uaux[0].reshape(TAM_IMG),cv.normalize(uaux[0], None, alpha=0, beta=1,norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F).reshape(TAM_IMG))
        # teste(u_t[0].reshape(TAM_IMG))

        

        # omega = cv.normalize(omega, None, alpha=0, beta=1,norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)
        # omega[0] = omega[0]*255.0
        # omega[0] = np.linalg.norm(omega[0])
        # print("VALOR DO OMEGA[0]: "+str(omega[0]))
        # descargo = []
        # for y in range(0,phi_aux.shape[0]):
        # for x in range(0,omega.shape[0]):
        #     norm_omega = np.linalg.norm(omega[x])
        #     omega[x] = omega[x]/norm_omega
        # for x in range(0,K):
            # print("VALOR DO OMEGA[0]["+str(x)+"]: "+str(omega[0][x]))
            # auxiliar = cv.normalize(aux_zero[x], None, alpha=0, beta=1,norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)
            # print("SHAPE DO AUXILIAR: "+str(auxiliar.shape))
            # auxiliar = auxiliar*255.0
            # auxiliar = np.linalg.norm(u_t[x])
            # print("VALOR DO OMEGA["+str(y)+"]["+str(x)+"]: "+str(omega[y][x]))
            # print("VALOR DO OMEGA[0]["+str(x)+"]: "+str(omega[0][x]))
            # aux_tst = (np.dot(omega[0][x],u_t[x])) + aux_tst
            # aux_tst = (np.dot(u_t[x],omega[0][x])) + aux_tst
            # aux_tst = (omega[y][x]*u_t[x]) + aux_tst
            # aux_tst = (omega[0][x]*aux_zero[x]) + aux_tst
            # aux_tst = aux_zero[x] + aux_tst
            # aux_tst = (omega[0][x]*aux_zero[x]) + aux_tst
            # aux_tst = (omega[0][x]*u_t[x]) + aux_tst
            # aux_tst = (omega[0][x]*uteste[x]) + aux_tst
            # aux_tst = (omega[0][x]*aux_M[x]) + aux_tst
            # aux_tst = (omega[0][x]*u_bostao[x]) + aux_tst
            # descargo.append(omega[0][x]*u_t[x])
            # aux_tst = (omega[0][x]*auxiliar) + aux_tst
            # print("VALOR DO AUX_TST: "+str(aux_tst))
        # aux_tst = cv.normalize(aux_tst, None, alpha=0, beta=1,norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)
        # print("VALOR DO AUX_TST:" +str(aux_tst))
        # print("SHAPE DO AUX_TST: "+ str(aux_tst.shape))
        # aux_tst = aux_tst + psi
        # print("VALOR DO AUX_TST:" +str(aux_tst))
        
        # print("VALOR DO AUX_TST:" +str(aux_tst))
        # teste(aux_tst.reshape(TAM_IMG))
        # teste(((aux_tst*255.0)).reshape(TAM_IMG))
        # teste((aux_zero[0] + psi).reshape(TAM_IMG))
        # aux_tst = 0
        # print("SHAPE DO AUX_TST: "+ str(aux_tst.shape))
        # print("RESHAPE: "+str(aux_tst.reshape(TAM_IMG)))
        # descargo = np.array(descargo)
        # print("DESCARGO SHAPE: "+str(descargo))
        # seila = 0
        # for x in range(0,K):
        #     print(str(x)+" |VALOR DO SEILA ANTES:"+str(seila))
        #     print(str(x)+" |VALOR DO DESCARGO: "+str(descargo[x]))
        #     seila = descargo[x] + seila
        #     print(str(x)+" |VALOR DO SEILA DEPOIS:"+str(seila))
        # print(seila)
        # teste(seila.reshape(TAM_IMG))
        # print("DESCARGO[0]: "+str(descargo[0]))
        # teste(descargo.reshape(TAM_IMG))
        # aux_tst = aux_tst*255.0
        # print("RESHAPE2: "+str(aux_tst))
        # print(psi+np.transpose(aux_tst))
        # aux_tst = np.linalg.norm(aux_tst)
        # aux_tst = cv.normalize(aux_tst.reshape(TAM_IMG), None, alpha=0, beta=1,norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)
        # lol = (psi+np.transpose(aux_tst))*255.0
        # teste(lol.reshape(TAM_IMG))
        # teste((aux_tst*255.0).reshape(TAM_IMG))
        # teste((psi+np.transpose(aux_tst)).reshape(TAM_IMG))


        #FIM TESTE DE REPRESENTAÇÃO DAS EIGENFACES NO FACESPACE

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

        # eg_vle_T =np.array([eg_vec[0]]) 
        # aux_phi = np.transpose(np.array([phi[0]]))
        # w = np.dot(aux_phi,eg_vle_T)
        # print("VALORES QUE ESTOU TESTANDO")
        # eg_face = np.dot(w,eg_vec[0])
        # if (eg_face == phi[0]).all():
            # print("ESTA IGUAL")
        # else:
            # print("TA DIFERENTE")
        # teste(phi[0].reshape((300,300)))
        # teste(eg_face.reshape((300,300)))
        
        # w = []
        # for i in range(0,qnt_gamma):
        #     w.append(np.dot(eg_vle[i]))

        for x in range(0,omega.shape[0]):
            aux_tst = 0
            for y in range(0,K):
                # aux_tst = (omega[x][y]*aux_zero[y]) + aux_tst
                # aux_tst = (omega[x][y]*u_bostao[y]) + aux_tst
                teste2(u_bostao[y].reshape(TAM_IMG),np.dot(omega[x][y],u_bostao[y]).reshape(TAM_IMG))
                aux_tst = (np.dot(omega[x][y],u_bostao[y])) + aux_tst
                
            # aux_tst = psi + aux_tst
            
            aux_bosta = np.linalg.norm(aux_tst)
            aux_bosta = aux_tst/aux_bosta

            aux_psiquico = np.linalg.norm(psi)
            aux_psiquico = psi/aux_psiquico

            aux_psiquico = aux_psiquico*255.0
            aux_tst = aux_bosta*255.0

            aux_tst = aux_tst + aux_psiquico
            # aux_tst = aux_bosta/255.0
            
            # teste(aux_tst.reshape(TAM_IMG))
            # teste2((phi_aux[x]+psi).reshape(TAM_IMG),aux_tst.reshape(TAM_IMG))
            teste3((phi_aux[x]+psi).reshape(TAM_IMG),u_bostao[x].reshape(TAM_IMG),aux_tst.reshape(TAM_IMG))

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

    def save_psi(self,psi):
        client = db.conecta(self)
        GDA = gd.connect()
        psi = psi.tolist()
        nome_arq = "face_media"
        save_json = '{"PSI":'+str(psi)+'}'
        eigenfaces.img2json(self,nome_arq,save_json,GDA,client,nome_arq,"PSI")
    
    def save_cov(self,cov):
        # print("COV SHAPE: "+str(cov.shape))
        client = db.conecta(self)
        GDA = gd.connect()
        # np.savetxt("TESTE.csv", cov, delimiter=",")
        for x in range(0,K):
            cov_list = cov[x].tolist()
        # jteste = json.dumps(cov)
            nome_arq = "covariancia_"+str(x)
            save_json = '{"COV":'+str(cov_list)+'}'
            eigenfaces.img2json(self,nome_arq,save_json,GDA,client,nome_arq,"COV")


tst = eigenfaces()
# tst.db2img()
# tst.eigNovo()
# tst.eig_mod()
tst.get_eigenfaces()
# tst.testa_psi()
# tst.teste_cov()
# tst.teste_phi()
# tst.img2db()
