import tkinter as tk
from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk
from cv2 import *
from extract import *
from eigen import *
from faceDetection import *
import time
import os

window = tk.Tk()

# WINDOW CONFIGURATION
window.title("Face Recognition")
window.geometry("1280x720")
window.config(bg="#F1F3FF")
window.option_add("*Font", "Montserrat")
window.resizable(False, False)


# CONFIG
H1_tuple = ("Montserrat-Bold", 48)
H2_tuple = ("Montserrat-Medium", 20)
H2_new = ("Montserrat-Bold", 20)
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


class openDataSet:
    def combineFunc(*funcs):
        def combinedFunc(*args, **kwargs):
            for f in funcs:
                f(*args, **kwargs)

        return combinedFunc

    global strData
    strData = tk.StringVar()
    strData.set("")

    def openData():
        global pathFile, strData1, strData, matriksKovarian, names, extract, matriksSelisih, cov, eigValue, eigVector, eigenDataFace, weight, startData, intervalData
        pathFile = filedialog.askdirectory()
        tes = os.path.basename(pathFile)

        print(pathFile)
        try:
            strData.set("Succesfully choosed")

            startData = time.time()
            names, extract = batch_extractor(pathFile)
            matriksKovarian = covarian(extract)
            matriksSelisih = selisih(extract, mean(extract))
            eigValue, eigVector = qr_iteration(matriksKovarian)
            eigenDataFace = eigenFace(matriksSelisih, eigVector)
            weight = weightFace(eigenDataFace, matriksSelisih)

            print("Vektor eigen: ")
            print(eigVector)
            print(
                "-------------------------------------------------------------------------------------")
            print("Eigenface: ")
            print(eigenFace)
            print(
                "-------------------------------------------------------------------------------------")
            print("Weight Face: ")
            print(weight)

            endData = time.time()
            intervalData = endData - startData
            print("Interval process data: ")
            print(intervalData)

        except FileNotFoundError:
            strData.set("Dataset not chosen")

        selectedData = tk.Label(textvariable=strData,
                                bg=bg_color,
                                fg=main_color,
                                font=Body_tuple
                                )
        selectedData.place(x=100, y=260)

    global path, strImage, strResult
    strImage = tk.StringVar()
    strImage.set("")
    strResult = tk.StringVar()
    strResult.set("")

    def open_Image():
    
            
        global imagePath, getImage, displayed, end, output, displayedResult, interval, path
        imagePath = filedialog.askopenfilename(filetypes=[('Images JPG', "*.jpg")])
        path = os.path.basename(imagePath)
            
        try: 
            strImage.set(path)
            canvasImage = Canvas(window, width=256, height=256)
            canvasImage.pack()

            getImage = Image.open(imagePath)
            resize_displayed = getImage.resize((256, 256))
            displayed = ImageTk.PhotoImage(resize_displayed)
            canvas.create_image(530, 240, anchor=NW, image=displayed)

            start = time.time()
            query = extractFace(imagePath, extract)
            queryImageWeight = queryWeight(eigenDataFace, query)
            eucDistance = euclideanDistance(weight, queryImageWeight)
            minDistance, match = bestMatch(names, eucDistance)

            print(
                "-------------------------------------------------------------------------------------")
            print("Jarak euclidean paling kecil : " + str(minDistance))
            print("Best match: " + str(match))
            #src = "src/dataset/"
            output = str(match)
            print(
                "-------------------------------------------------------------------------------------")
            print("Hasil")
            print(output)
            end = time.time()

            print(intervalData)
            interval = intervalData + (end - start)
            print(interval)

            rslt = str(pathFile) + "/" + output
            print(rslt)

            openResult = Image.open(rslt)
            resizeResult = openResult.resize((256, 256))
            displayedResult = ImageTk.PhotoImage(resizeResult)

            frame = Frame(window, width=256, height=256)
            frame.pack()
            frame.place(anchor=NW, relx=0.75, rely=0.33)

            labelResult = tk.Label(
                frame, image=displayedResult, borderwidth=0, highlightthickness=0)
            labelResult.pack()

            strResult.set(output)
            displayresult = tk.Label(window, textvariable=strResult,
                                     font=Body_tuple,
                                     bg=bg_color,
                                     fg=main_color)
            displayresult.place(x=100, y=600)

            rundown = tk.Label(text=interval,
                               font=Body_tuple,
                               bg=bg_color,
                               fg="#FF0000")
            rundown.place(x=700, y=550)
        except AttributeError:
            strImage.set("No file chosen")
        except NameError:
            strImage.set("Please choose dataset first")

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

    buttonFileOpen = PhotoImage(file=f'{img_dir}/src/components/button1.png')
    buttonImageOpen = tk.Button(
        font=Body_tuple,
        bd=0,
        image=buttonFileOpen,
        bg=bg_color,
        command=combineFunc(open_Image),
    )
    buttonImageOpen.place(x=100, y=410)

    executionTime = tk.Label(text="Execution time: ",
                             font=Body_tuple,
                             bg=bg_color,
                             fg=main_color)
    executionTime.place(x=500, y=550)

    inputFile = tk.Label(text="Input Your Dataset",
                         font=H2_tuple,
                         fg=main_color,
                         bg=bg_color)
    inputFile.place(x=100, y=180)

    img_dir = os.getcwd()

    buttonFileImage = PhotoImage(file=f'{img_dir}/src/components/button2.png')
    buttonFile = tk.Button(
        bd=0,
        bg=bg_color,
        font=Body_tuple,
        image=buttonFileImage,
        command=openData
    )
    buttonFile.place(x=100, y=230)

    def openCamera():
        global img_item, getImage, displayed, labelResult, openResult,  displayedResult, frame
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')
        cancel = False

        cap = cv2.VideoCapture(0)

        dirname = os.path.dirname(__file__)
        pathname = dirname.replace("src", "")

        startCamera = time.time()

        while (True):
            global cam, imagePath1, imagePath2

            ret, frame = cap.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
            faces = face_cascade.detectMultiScale(
                gray, scaleFactor=1.5, minNeighbors=5)
            displayCam = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)

            for (x, y, w, h) in faces:
                roi_gray = gray[y:y+h, x:x+w]
                roi_color = frame[y:y+h, x:x+w]
                img_item = "./src/test/captureNow.png"
                cv2.imwrite(img_item, displayCam)

            endCamera = time.time()

            if (endCamera - startCamera >= 7):
                break
            cv2.imshow('', frame)
            if cv2.waitKey(20) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

          
        try:
            pathCamera = os.path.basename(img_item)
            strImage.set(pathCamera)
            canvasImage = Canvas(window, width=256, height=256)
            canvasImage.pack()

            getImage = Image.open(img_item)
            resize_displayed = getImage.resize((256, 256))
            displayed = ImageTk.PhotoImage(resize_displayed)
            canvas.create_image(530, 240, anchor=NW, image=displayed)

            start = time.time()

            queryCamera = extractFace(img_item, extract)
            queryWeightCamera = queryWeight(eigenDataFace, queryCamera)
            eucDistanceCamera = euclideanDistance(weight, queryWeightCamera)
            minDistanceCam, matchCam = bestMatch(names, eucDistanceCamera)

            print(
                "-------------------------------------------------------------------------------------")
            print("Jarak paling kecil: " + str(minDistanceCam))
            print(str(matchCam))
            output = str(matchCam)
            print(
                "-------------------------------------------------------------------------------------")
            print("Hasil")
            print(output)
            end = time.time()

            print(intervalData)
            interval = intervalData + (end - start)
            print(interval)

            rslt = str(pathFile) + "/" + output
            print(rslt)

            strResult.set(output)

            openResult = Image.open(rslt)
            resizeResult = openResult.resize((256, 256))
            displayedResult = ImageTk.PhotoImage(resizeResult)

            frame = Frame(window, width=256, height=256)
            frame.pack()
            frame.place(anchor=NW, x=0.75, rely=0.32)

            labelResult = tk.Label(
                frame, image=displayedResult, borderwidth=0, highlightthickness=0)
            labelResult.pack()

            displayresult = tk.Label(textvariable=strResult,
                                     font=Body_tuple,
                                     bg=bg_color,
                                     fg=main_color)
            displayresult.place(x=100, y=600)

            rundown = tk.Label(text=interval,
                               font=Body_tuple,
                               bg=bg_color,
                               fg="#FF0000")

            rundown.place(x=700, y=550)
            
        except NameError:
            strImage.set("Oops cannot detect face")
            
            
    buttonCameraImage = PhotoImage(
        file=f'{img_dir}/src/components/button3.png')
    buttonCamera = tk.Button(
        bd=0,
        font=Body_tuple,
        bg=bg_color,
        image=buttonCameraImage,
        command=openCamera,
    )
    buttonCamera.place(x=100, y=500)

    global buttonUseOpen
    img_dir = os.getcwd()

    def howToUse():
        global useOne, useTwo, useThree
        useWindow = Toplevel(window)
        useWindow.geometry("850x300")
        useWindow.title("How To Use Our Face Recognition")
        useWindow.config(bg=bg_color)

        useTitle_text = tk.Label(useWindow, text="How To Use",
                                 font=H2_new,
                                 fg=main_color,
                                 bg=bg_color)
        useTitle_text.place(x=355, y=20)

        img_dir = os.getcwd()
        useOne = PhotoImage(file=f'{img_dir}/src/components/image1.png')
        useTwo = PhotoImage(file=f'{img_dir}/src/components/image2.png')
        useThree = PhotoImage(file=f'{img_dir}/src/components/image3.png')

        displayedOne = Label(useWindow, image=useOne, bg=bg_color)
        displayedOne.place(x=35, y=70)

        displayedTwo = Label(useWindow, image=useTwo, bg=bg_color)
        displayedTwo.place(x=325, y=70)

        displayedThree = Label(useWindow, image=useThree, bg=bg_color)
        displayedThree.place(x=600, y=70)

    buttonUseOpen = PhotoImage(file=f'{img_dir}/src/components/button4.png')
    buttonUse = tk.Button(
        font=Body_tuple,
        bd=0,
        bg=bg_color,
        image=buttonUseOpen,
        command=howToUse,
    )
    buttonUse.place(x=120, y=100)


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
