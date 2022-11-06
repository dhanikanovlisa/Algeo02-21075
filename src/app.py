import tkinter as tk
import os
from tkinter import *
from tkinter import filedialog
from tkinter.filedialog import askopenfile
from PIL import Image, ImageTk

window = tk.Tk()

#WINDOW CONFIGURATION
window.title("Face Recognition")
window.geometry("1920x1024")
window.config(bg="#F1F3FF")
window.option_add("*Font", "Montserrat")


#CONFIG
H1_tuple = ("Montserrat-Bold", 48)
H2_tuple = ("Montserrat-Medium", 20)
Body_tuple = ("Montserrat-Medium", 16)
bg_color = ("#F1F3FF")
main_color = ("#001666")
secondary_color = ("#B5C1EC")
third_color = ("#F5F5F5")

canvas = Canvas(window, height= 1024, width= 1920, bg=bg_color)
canvas.pack()

#TITLE
title_Menu = tk.Label(text="FACE RECOGNITION",
                      font= H1_tuple,
                      fg = main_color,
                      bg = bg_color)
title_Menu.place(x = 350, y = 40)

canvas.create_line(98, 150, 1246, 150, fill=main_color, width=2)
canvas.create_rectangle(1250, 180, 486, 550, outline="", fill=secondary_color)

strImage = tk.StringVar()
strImage.set("")
def open_Image():
    global img, img_resized
    image = filedialog.askopenfilename(filetypes=[('Images JPG', "*.jpg")])
    if image:
        strImage.set(image)
        fob = open(image, 'r')
        print(fob.read())
        img = Image.open(image)
        img_resized = img.resize((256,256))
        img = ImageTk.PhotoImage(img_resized)
    else:
        strImage.set("No file chosen")


#INPUT FILE
inputFile = tk.Label(text="Input Your Dataset",
                     font=H2_tuple,
                     fg = main_color,
                     bg = bg_color)
inputFile.place(x=100, y= 180)

buttonFile = tk.Button(text="Choose File",
                       width= 10,
                       font=Body_tuple,
                       )
buttonFile.place(x=100, y=230)

selectedFolder = tk.Label(
                          font=Body_tuple,
                          bg = bg_color,
                          fg=main_color)
selectedFolder.place(x=100, y=300)

inputImage = tk.Label(text="Input Your Image",
                      font = H2_tuple,
                      fg = main_color,
                      bg = bg_color
                      )
inputImage.place(x=100, y= 350)

selectedImage = tk.Label(textvariable=strImage,
                        bg= bg_color,
                         fg = main_color,
                         font = Body_tuple)
selectedImage.place(x=100, y= 470)

buttonImage = tk.Button(text="Choose File",
                        font = Body_tuple,
                        width = 10,
                        command = lambda:open_Image()
                        )
buttonImage.place(x=100, y=410)


text_Result = tk.Label(text = "Result",
                       font = H2_tuple,
                       bg = bg_color,
                       fg = main_color)
text_Result.place(x=100, y= 550)

testImage = tk.Label(text = "Test Image",
                    font = Body_tuple,
                    bg= secondary_color,
                    fg= main_color)
testImage.place(x= 550, y = 200)


imageResult = tk.Label(text = "Closest Result",
                    font = Body_tuple,
                    bg= secondary_color,
                    fg= main_color)
imageResult.place(x= 950, y = 200)

window.mainloop()