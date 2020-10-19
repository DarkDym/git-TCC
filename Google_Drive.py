from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

class connect_drive:
    def connect():
        gauth = GoogleAuth()
        gauth.LocalWebserverAuth()
        # gauth.CommandLineAuth()
        drive = GoogleDrive(gauth)
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

drive = connect_drive
drive.connect()