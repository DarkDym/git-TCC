from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
import json
import numpy as np

class connect_drive:
    def connect():
        gauth = GoogleAuth()
        gauth.LoadCredentialsFile("credentials.json")
        return gauth
        
    def get_files():
        GA = connect_drive.connect()
        drive = GoogleDrive(GA)
        fileList = drive.ListFile({'q': "'root' in parents and trashed=false"}).GetList()
        # print(fileList)
        for files in fileList:
            if (files['title'] == "IMAGES_TCC"):
                fileID = files['id']
                break
        fileList = drive.ListFile({'q': "'"+fileID+"' in parents and trashed=false"}).GetList()
        for files in fileList:
            print(files['title'])
            if (files['title'] == "teste2.json"):
                # down_file = drive.CreateFile({'id':files['id']})
                # down_file.GetContentFile('95-11.jpg')
                cont_file = drive.CreateFile({'id':files['id']})
                info = cont_file.GetContentString()
                print(info)
                break
    def get_json2img(GA,erro):
        # GA = connect_drive.connect()
        drive = GoogleDrive(GA)
        fileList = drive.ListFile({'q': "'root' in parents and trashed=false"}).GetList()
        # print(fileList)
        for files in fileList:
            if (files['title'] == "IMAGES_TCC"):
                fileID = files['id']
                break
        fileList = drive.ListFile({'q': "'"+fileID+"' in parents and trashed=false"}).GetList()
        # x = 1
        for files in fileList:
            # print(files['title'])
            # if (files['title'] == "pessoa"+str(x)+".json"):
                # down_file = drive.CreateFile({'id':files['id']})
                # down_file.GetContentFile('95-11.jpg')
            cont_file = drive.CreateFile({'id':files['id']})
            info = cont_file.GetContentString()
            tst = json.loads(info)
            print(type(tst["OMEGA"]))
            aux = np.array(tst["OMEGA"])
            print(type(aux))
            print(type(erro))
            if (erro == aux).all():
                print("OMEGA DO DRIVE: "+str(tst["OMEGA"]))
                print("NOME DO ARQUIVO: "+str(files['title']))
                return tst
                # info = cont_file.GetContentFile(files['title'])
                # print(tst["OMEGA"])
            # x += 1

    def get_credentials():
        gauth = GoogleAuth()
        gauth.LocalWebserverAuth()
        print("Arquivo credentials.json criado.")
    
    def create_img2json(name_arq,cont_arq,GA):
        # GA = connect_drive.connect()
        drive = GoogleDrive(GA)
        fileList = drive.ListFile({'q': "'root' in parents and trashed=false"}).GetList()
        
        for files in fileList:
            if (files['title'] == "IMAGES_TCC"):
                fileID = files['id']
                break
        
        fileSave = drive.CreateFile({'title': name_arq+str(".json"),'parents':[{'id':fileID}]})
        fileSave.SetContentString(str(cont_arq))
        fileSave.Upload()
        print("Arquivo criado com sucesso, ID: "+str(fileSave['id']))
        return fileSave['id']

# drive = connect_drive
# drive.connect()
# drive.get_files()