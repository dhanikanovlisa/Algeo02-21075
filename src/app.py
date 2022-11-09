import tkinter as tk
import os
from tkinter import *
from tkinter import filedialog
from tkinter.filedialog import askopenfile
from PIL import Image, ImageTk
import cv2
from cv2 import * 

window = tk.Tk()

# WINDOW CONFIGURATION
window.title("Face Recognition")
window.geometry("1920x1024")
window.config(bg="#F1F3FF")
window.option_add("*Font", "Montserrat")


# CONFIG
H1_tuple = ("Montserrat-Bold", 48)
H2_tuple = ("Montserrat-Medium", 20)
Body_tuple = ("Montserrat-Medium", 16)
bg_color = ("#F1F3FF")
main_color = ("#001666")
secondary_color = ("#B5C1EC")
third_color = ("#F5F5F5")

canvas = Canvas(window, height=1024, width=1920, bg=bg_color)
canvas.pack()

# TITLE
title_Menu = tk.Label(text="FACE RECOGNITION",
                      font=H1_tuple,
                      fg=main_color,
                      bg=bg_color)
title_Menu.place(x=350, y=40)

canvas.create_line(98, 150, 1246, 150, fill=main_color, width=2)
canvas.create_rectangle(1250, 180, 486, 550, outline="", fill=secondary_color)


class openImage():
    def combineFunc(*funcs):
        def combinedFunc(*args, **kwargs):
            for f in funcs:
                f(*args, **kwargs)

        return combinedFunc

    global path, strImage
    strImage = tk.StringVar()
    strImage.set("")
    

    
    def open_Image():
        global getImage, displayed
        image = filedialog.askopenfilename(filetypes=[('Images JPG', "*.jpg")])
        path = os.path.basename(image)
        if image:
            strImage.set(path)
            canvasImage = Canvas(window, width=256, height = 256)
            canvasImage.pack()
        
            getImage= Image.open(image)
            resize_displayed= getImage.resize((256, 256))
            displayed = ImageTk.PhotoImage(resize_displayed)
            canvas.create_image(530, 240, anchor = NW, image=displayed)
        else:
            canvas.create_image(530, 240, anchor = NW, image=None)
            strImage.set("No file chosen")
            
        
    inputImage = tk.Label(text="Input Your Image",
                          font=H2_tuple,
                          fg=main_color,
                          bg=bg_color
                          )

    inputImage.place(x=100, y=350)

    selectedImage = tk.Label(textvariable=strImage,
                             bg=bg_color,
                             fg=main_color,
                             font=Body_tuple)
    selectedImage.place(x=100, y=470)

    buttonImage = tk.Button(text="Choose File",
                            font=Body_tuple,
                            width=10,
                            command=combineFunc(open_Image),
                            )
    buttonImage.place(x=100, y=410)


class openDataSet():
    def combineFunc(*funcs):
        def combinedFunc(*args, **kwargs):
            for f in funcs:
                f(*args, **kwargs)

        return combinedFunc
    
    
    def openData():
        pathFile = filedialog.askdirectory()
        global strData1, strData
        print(pathFile)
        if pathFile:
            strData1 = tk.Label(text="Succesfully choosed", font=Body_tuple, fg=main_color, bg=bg_color)
            strData1.place(x=100, y= 300)
        else:
            strData = tk.Label(text="Dataset not chosen", font=Body_tuple, fg=main_color, bg=bg_color)
            strData.place(x=100, y= 300)
        

    inputFile = tk.Label(text="Input Your Dataset",
                         font=H2_tuple,
                         fg=main_color,
                         bg=bg_color)
    inputFile.place(x=100, y=180)
    
            
    buttonFile = tk.Button(text="Choose File",
                           width=10,
                           font=Body_tuple,
                           command=combineFunc(openData)
                           )
    buttonFile.place(x=100, y=230)
    

    
text_Result = tk.Label(text="Result",
                       font=H2_tuple,
                       bg=bg_color,
                       fg=main_color)
text_Result.place(x=100, y=550)

testImage = tk.Label(text="Test Image",
                     font=Body_tuple,
                     bg=secondary_color,
                     fg=main_color)
testImage.place(x=550, y=200)


imageResult = tk.Label(text="Closest Result",
                       font=Body_tuple,
                       bg=secondary_color,
                       fg=main_color)
imageResult.place(x=950, y=200)

executionTime = tk.Label(text="Execution time: ",
                         font=Body_tuple,
                     bg=bg_color,
                     fg=main_color)
executionTime.place(x= 500, y = 550)

window.mainloop()


#tinggal print file directory atau gausah
#place foto
#file image dan directory connect ke fungsi extract feature
#execusion time
