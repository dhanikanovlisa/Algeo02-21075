import tkinter as tk
import os
from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk
from cv2 import * 
from extract import *
from eigen import *
import time

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


class openImage:
    def combineFunc(*funcs):
        def combinedFunc(*args, **kwargs):
            for f in funcs:
                f(*args, **kwargs)

        return combinedFunc

    


class openDataSet:
    def combineFunc(*funcs):
        def combinedFunc(*args, **kwargs):
            for f in funcs:
                f(*args, **kwargs)

        return combinedFunc
    
    

    global path, strImage
    strImage = tk.StringVar()
    strImage.set("")
    
    def open_Image():
        global imagePath, getImage, displayed, end, output, displayedResult, interval
        imagePath = filedialog.askopenfilename(filetypes=[('Images JPG', "*.jpg")])
        path = os.path.basename(imagePath)
        if imagePath:
            strImage.set(path)
            canvasImage = Canvas(window, width=256, height = 256)
            canvasImage.pack()
        
            getImage= Image.open(imagePath)
            resize_displayed= getImage.resize((256, 256), Image.LANCZOS)
            displayed = ImageTk.PhotoImage(resize_displayed)
            canvas.create_image(530, 240, anchor = NW, image=displayed)
            
            start = time.time()
            query = extract_features(imagePath)
            queryArray = []
            queryArray.append(query)
                
            querySelisih = selisih(queryArray, mean(extract))
            queryWeight = np.matmul(np.transpose(face), np.transpose(querySelisih))
            distance = np.linalg.norm(weight - queryWeight, axis = 0)
            bestMatch = np.argmin(distance)
            print("-------------------------------------------------------------------------------------")
            print(distance)
            print(names[bestMatch])
                #src = "src/dataset/"
            output = str(names[bestMatch])
            print("-------------------------------------------------------------------------------------")
            print("Hasil")
            print(output)
            end = time.time()
            
            print(intervalData)
            interval = intervalData + (end - start)
            print(interval)
            
            rslt = str(pathFile) + "/" + output
            print(rslt)
            
            openResult = Image.open(rslt)
            resizeResult = openResult.resize((256,256), Image.LANCZOS)
            displayedResult = ImageTk.PhotoImage(resizeResult)
            
            frame = Frame(window, width=256, height=256, borderwidth=0, highlightthickness=0)
            frame.pack()
            frame.place(anchor=NW, relx=0.68, rely=0.32)
            
            labelResult = tk.Label(frame, image=displayedResult)
            labelResult.pack()
            
            
            displayresult = tk.Label(text = output,
                                     font=Body_tuple,
                            bg=bg_color,
                            fg=main_color)
            displayresult.place(x=100, y=600)
            
            
            rundown = tk.Label(text = interval,
                            font=Body_tuple,
                            bg=bg_color,
                            fg="#FF0000")
            rundown.place(x =700, y = 550)
        
        
        else:
            canvas.create_image(530, 240, anchor = NW, image=None)
            strImage.set("No file chosen")
        return imagePath
            
        
        
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
    selectedImage.place(x=100, y=440)
    
    img_dir = os.getcwd()

    
        
    buttonFileOpen = PhotoImage(file=f'{img_dir}/src/image/button1.png')
    buttonImageOpen = tk.Button(
                            font=Body_tuple,
                            bd = 0,
                            image = buttonFileOpen,
                            command=combineFunc(open_Image),
                            )
    buttonImageOpen.place(x=100, y=410)
    
    
    
    def openData():
        global pathFile, strData1, strData, cov, names, extract, matSelisih, cov, eigVal, eigVec, face, weight, startData, intervalData
        pathFile = filedialog.askdirectory() 
        tes = os.path.basename(pathFile) 
        print(pathFile)
        print(tes)
        if pathFile:
            strData1 = tk.Label(text="Succesfully choosed", font=Body_tuple, fg=main_color, bg=bg_color)
            strData1.place(x=100, y= 260)
            
            startData = time.time()
            names, extract = batch_extractor(pathFile)
            matSelisih = selisih(extract, mean(extract))
            cov = covarian(extract)
            eigVal, eigVec = qr_iteration(cov)
            face = eigenFace(matSelisih, eigVec)
            weight = weightFace(face, matSelisih)
            print("Vektor eigen: ")
            print(eigVec)
            print("-------------------------------------------------------------------------------------")
            print("Eigenface: ")
            print(face)
            print("-------------------------------------------------------------------------------------")
            print("Weight Face: ")
            print(weight)
            
            endData = time.time()
            intervalData = endData - startData
            print("Interval process data: ")
            print(intervalData)
            
        
            
            #resultImage = Canvas(window, width =256, height = 256)
            #resultImage.pack()
            
            #getImage = Image.open(file)
            #closest_displayed = getImage.resize((256,256), Image.LANCZOS)
            #displayed = ImageTk.PhotoImage(closest_displayed)
            #canvas.create_image(600, 240, anchor = NW, image= displayed)
                
        else:
            strData = tk.Label(text="Dataset not chosen", font=Body_tuple, fg=main_color, bg=bg_color)
            strData.place(x=100, y= 300)
    
    executionTime = tk.Label(text="Execution time: ",
                         font=Body_tuple,
                     bg=bg_color,
                     fg=main_color)
    executionTime.place(x= 500, y = 550)      
            

            
    
    
    inputFile = tk.Label(text="Input Your Dataset",
                         font=H2_tuple,
                         fg=main_color,
                         bg=bg_color)
    inputFile.place(x=100, y=180)
    
    img_dir = os.getcwd()
    
    buttonFileImage = PhotoImage(file=f'{img_dir}/src/image/button2.png')  
    buttonFile = tk.Button(
                           bd = 0,
                           font=Body_tuple,
                           image = buttonFileImage,
                           command=openData
                           )
    buttonFile.place(x=100, y=230)
    
class openCamera:
    def combineFunc(*funcs):
        def combinedFunc(*args, **kwargs):
            for f in funcs:
                f(*args, **kwargs)

        return combinedFunc
    
     
    def showFrame():
        newWindow = Toplevel(window)
        newWindow.title("Camera")
        newWindow.geometry("500x500")
        
        cap= cv2.VideoCapture(0)
    
        def capture():
            
            cv2image= cv2.cvtColor(cap.read()[1],cv2.COLOR_BGR2RGB)
            img = Image.fromarray(cv2image)
            imgtk = ImageTk.PhotoImage(image = img)
            newWindow.imgtk = imgtk
            newWindow.configure(image=imgtk)
            # Repeat after an interval to capture continiously
            newWindow.after(20, capture)
        
        capture()
    
    img_dir = os.getcwd()
    
    buttonCameraImage = PhotoImage(file=f'{img_dir}/src/image/button3.png')  
    buttonCamera = tk.Button(
                           bd = 0,
                           font=Body_tuple,
                           image = buttonCameraImage,
                           command = showFrame,
                           )
    buttonCamera.place(x= 100, y = 500)
    
    
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






def run():
    window.mainloop()

run()
#tinggal print file directory atau gausah
#place foto
#file image dan directory connect ke fungsi extract feature
#execusion time
