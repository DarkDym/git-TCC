from tkinter import *
import numpy as np
from PIL import Image, ImageTk
import time
import cv2 as cv
from matplotlib import pyplot as plt

#-----------Importações Próprias-------------------
from Database import db_mongo as db
from Frequency import GeraPDF as pdf
from HaarLike import harr as hr
from Eigen import eigenfaces as eg
#-----------Importações Próprias-------------------

WIDTH = 640
HEIGHT = 480

class Application:
    def __init__(self, window, master=None):
        #db.conecta(self)
        #pdf.tabela(self,"teste")
        print("OI")
        # eg.classififcca(self)
        self.window = window

        self.cam = cv.VideoCapture(0)
        #self.cam = cv.VideoCapture(0,cv.CAP_DSHOW)
        self.cam.set(cv.CAP_PROP_FRAME_WIDTH,WIDTH)
        self.cam.set(cv.CAP_PROP_FRAME_HEIGHT,HEIGHT)
        #self.cam.set(cv.CAP_PROP_FORMAT, cv.COLOR_BGR2GRAY)
        frm = self.cam.read()

        self.master = Frame(master)
        self.master.pack()

        root.protocol("WM_DELETE_WINDOW",self.on_closing)

        self.cnvs_cont = Canvas(master, width=WIDTH, height=HEIGHT)
        self.cnvs_cont.pack(side=LEFT)

        self.form = Frame(master)
        self.form.pack()

        self.label_master1 = Frame(master)
        self.label_master1.pack()

        self.label_nome = Label(self.label_master1, text="Nome:",font=('Verdana','12'))
        self.label_nome.pack(side=LEFT)

        self.form_nome = Entry(self.label_master1, width=25)
        self.form_nome.insert(0,"NOME")
        self.form_nome.config(state="disabled")
        self.form_nome.pack(side=RIGHT)

        self.label_master2 = Frame(master)
        self.label_master2.pack()

        self.label_sem = Label(self.label_master2, text="Semestre:",font=('Verdana','12'))
        self.label_sem.pack(side=LEFT)

        self.form_sem = Entry(self.label_master2, width=25)
        self.form_sem.insert(0,"10º")
        self.form_sem.config(state="disabled")
        self.form_sem.pack(side=RIGHT)

        self.btn_cont1 = Frame(master)
        self.btn_cont1.pack()

        self.btn_foto = Button(self.btn_cont1)
        self.btn_foto['text'] = "Foto"
        self.btn_foto['font'] = ("Calibri","10")
        self.btn_foto['width'] = "12"
        self.btn_foto['command'] = self.tira_foto
        self.btn_foto.pack()

        self.btn_foto2 = Button(self.btn_cont1)
        self.btn_foto2['text'] = "Foto2"
        self.btn_foto2['font'] = ("Calibri","10")
        self.btn_foto2['width'] = "12"
        self.btn_foto2['command'] = self.tira_foto2
        self.btn_foto2.pack()

        self.sc_cam() #Com cam descomenta | sem cam comenta

    def sc_cam(self):
        self.update()
        self.window.mainloop()

    def tira_foto(self):
        #cam = cv.VideoCapture(0)
        #cam.set(cv.CAP_PROP_FRAME_WIDTH,300)
        #cam.set(cv.CAP_PROP_FRAME_HEIGHT,400)
        #frm = cam.read()
        #tst = cv.cvtColor(frm[1], cv.COLOR_BGR2GRAY)
        #print(tst)
        #img = cv.line(img, (0,0), (400,400), (255,0,0), 1)
        tst = self.cam.read()
        cv.imwrite("frame-" + time.strftime("%d-%m-%Y-%H-%M-%S") + ".jpg", cv.cvtColor(tst[1], cv.COLOR_BGR2GRAY))
        self.photo = ImageTk.PhotoImage(image = Image.fromarray(tst[1]))
        mx = WIDTH/2
        my = HEIGHT/2
        self.cnvs_cont.create_image(mx,my, image=self.photo, anchor=NW)
        print("BTN APERTADO")
        #cam.release()
    def tira_foto2(self):
        #img = cv.imread("C://Users//tcc//Documents//img1.jpg",0)
        frame_tst = self.cam.read()
        img = cv.cvtColor(frame_tst[1], cv.COLOR_BGR2GRAY)
        #img = cv.resize(frame_tst[1],(300,400))
        mx = WIDTH/4
        my = HEIGHT/10
        [x,y] = img.shape
        hist = np.zeros(256)
        #mask = np.array([[-1,1,-1],[1,8,1],[-1,1,-1]])
        #img2 = cv.filter2D(img,cv.CV_64F,mask) 
        for i in range(x):
            for j in range(y):
                hist[img[i][j]] = hist[img[i][j]] + 1
        
        histN = np.zeros(256)
        for x in range(np.size(hist)):
            for y in range(x):
                histN[x] = hist[y] + histN[x]

        #print(hist)
        plt.plot(hist)
        plt.plot(histN)          
        #img2 = cv.Laplacian(img,cv.CV_64F)
        #sobelx = cv.Sobel(img,cv.CV_64F,1,0,ksize=1)
        #sobely = cv.Sobel(img,cv.CV_64F,0,1,ksize=1)
        #sobel = sobelx + sobely
        #img = cv.line(img, (0,0), (400,400), (255,0,0), 1)
        self.photo = ImageTk.PhotoImage(image = Image.fromarray(img))
        self.cnvs_cont.create_image(mx,my, image=self.photo, anchor=NW)
        print("BTN APERTADO")
        plt.show()

    def update(self):
        ret, frm = self.cam.read()
        tst = cv.cvtColor(frm, cv.COLOR_BGR2GRAY) #Com cam descomenta | sem cam comenta 
        if ret :
            frmhr = hr.detecta(self,frm)
            #self.photo = ImageTk.PhotoImage(image = Image.fromarray(tst))
            self.photo = ImageTk.PhotoImage(image = Image.fromarray(frmhr))
            self.cnvs_cont.create_image(0,0, image=self.photo, anchor=NW)
        self.window.after(15, self.update)

    def on_closing(self):
        print("MORREU")
        self.cam.release()
        root.destroy()   

root = Tk()
Application(root)
root.mainloop()
