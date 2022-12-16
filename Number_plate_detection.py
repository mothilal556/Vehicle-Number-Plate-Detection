import cv2
import numpy as np
import imutils
import pytesseract
from csv import writer
from datetime import datetime

pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

plateCascade = cv2.CascadeClassifier("haarcascade_russian_plate_number.xml")
minArea = 300

cap =cv2.VideoCapture("1.mp4")

count = 0

while cap.isOpened():
    success , img  = cap.read()
    img = imutils.resize(img,width=320)
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    numberPlates = plateCascade .detectMultiScale(imgGray, 1.1, 4)

    for (x, y, w, h) in numberPlates:
        wT, hT, cT = img.shape
        a, b = (int(0.02 * wT), int(0.02 * hT))
        plate = img[y + a:y + h - a, x + b:x + w - b, :]
        # make the img more darker to identify LPR
        kernel = np.ones((1, 1), np.uint8)
        plate = cv2.dilate(plate, kernel, iterations=1)
        plate = cv2.erode(plate, kernel, iterations=1)
        plate_gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
        (thresh, plate) = cv2.threshold(plate_gray, 127, 255, cv2.THRESH_BINARY)
        # read the text on the plate
        read = pytesseract.image_to_string(plate)
        read = ''.join(e for e in read if e.isalnum())
        stat=read[0:2]
        cv2.rectangle(img,(x,y),(x+w,y+h),(51,51,255),2)
        cv2.rectangle(img,(x-1,y-40),(x+w+1,y),(51,51,255),-1)
        cv2.putText(img,read,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.9,(255,255,255),2)

        if read!="":
            now = datetime.now()
            dt_string = now.strftime("%d-%m-%Y %H:%M:%S")
            print( dt_string, read)
            List = [dt_string,read]
            with open('Vehicle_details.csv', 'a') as f_object:
                writer_object = writer(f_object)
                writer_object.writerow(List)
                f_object.close()
            now = datetime.now()
            dt_string = now.strftime("%d-%m-%Y %H%M%S")
            imgRoi = img[y:y + h, x:x + w]
            cv2.imshow("ROI", imgRoi)
            cv2.imwrite("Output_Images/IMAGES_" +dt_string+ ".jpg", imgRoi)

    cv2.imshow("Result",img)
    if cv2.waitKey(1) & 0xFF ==ord('q'):
        cv2.imshow("Result",img)
        cv2.waitKey(500)
        break
