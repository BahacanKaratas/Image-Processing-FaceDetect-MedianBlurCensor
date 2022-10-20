import numpy as np
import cv2
#import face_recognition
#from simple_facerec import SimpleFacerec
"""
This is my median blur function, which we discussed at class, I developed myself by researching trough internet and friends to achive
the fastest algorithm. This is necessarry since I will run it 4 times. The size of 5x5 matrix is (as my research showed) the best possible
size.
"""
def myMedianBlur(image: np.ndarray):
    """
    Do not change the original
    """
    copiedImg = np.copy(image)
    """
    Making borders larger since a 5x5 matrix needs 2 additional pixels to achive median blur
    """
    altered = cv2.copyMakeBorder(copiedImg , 2, 2, 2, 2, cv2.BORDER_REPLICATE)
    x, y, z = copiedImg.shape
    """
    Setting up the values varible to extract the median after filling it with pixels data
    """
    values = [
        (i, j)
        for i in range(-2,3)
        for j in range(-2,3)
    ]
    median = len(values) // 2
    """
    The median blur algorithm , implementation of the median blur filter.
    """
    for i in range(x):
        for j in range(y):
            for u in range(z):
                copiedImg[i][j][u] = sorted(altered[i+ 2 +a][j + 2 + b][u] \
            for a,b in values)[median]
    """
    Return the  blurred img
    """
    return copiedImg
"""
More information about haar cascades and features will be discussed in the demo.

"""
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

"""
Read the img
"""
img = cv2.imread("images/foto5.jpg")
cv2.imshow('Un-edited', img)

"""
To detect faces, I need to aply cvtColor method.
"""
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.3, 5)
"""
I need this arrays since I will crop out individual faces one by one in the next section so this is my way
of keeping track of where the faces are
"""
startx=[];starty=[];endx=[];endy=[]
for (x,y,w,h) in faces:
    """
    Add points to my tracking arrays
    """
    startx.append(x)
    starty.append(y)
    endx.append(w)
    endy.append(h)
    
    """
    This is a loop where it puts a rectangle over a detected face using x,y,w,h which are the positions
    which relate to the location(pixels) of the face we recently detected
    """    
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    
cv2.imshow('FaceDetected', img)
"""
In this part, I import a varible from a library I found and implemented into my project.
What it does is it takes images/ folder as a trained set to identify unknown faces, by matching them to the images in this folder.
"""
"""
#FACE REC-----------------------------------------------------------------------------------------------------------------------
sfr = SimpleFacerec()
sfr.load_encoding_images("images/")
#FACE REC-----------------------------------------------------------------------------------------------------------------------
"""
"""
Here it uses detect known faces method which in the simple_facerec.py file defined to run trough faces features and try to match them
to the faces we know (which are in the images/ folder and encoded).
If they are matched, the name of the photo is displayed as the known face.
"""
"""
#Face Rec-----------------------------------------------------------------------------------------------------------------------------
frame=img
#Face Rec
#FACE REC
face_locations, face_names = sfr.detect_known_faces(frame)
for face_loc, name in zip(face_locations, face_names):
    y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]
#FACE REC
cv2.putText(frame, name,(x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)
#FACE REC------------------------------------------------------------------------------------------------------------------------------
"""
"""
As I collected individual faces location points, with the help of this loop, I can extract the faces, apply my median blur to them
for 4 times to censor them (by blurring enough but not so much that it disrupts the image as a whole) and return this cropped faces
to original img and show it as a final censorised image.
"""
for i in range (len(startx)):
     #crop the faces 
    croppedimg=img[starty[i]:starty[i]+endy[i] , startx[i]:startx[i]+endx[i]]
    #apply median blur 4 times (this is why I tried my best to implement a fast median blur)
    tempImg1=myMedianBlur(croppedimg)
    tempImg2=myMedianBlur(tempImg1)
    tempImg3=myMedianBlur(tempImg2)
    tempImg4=myMedianBlur(tempImg3)
    #put the censored faces back
    img[starty[i]:starty[i]+endy[i] , startx[i]:startx[i]+endx[i]]=myMedianBlur(tempImg4)
        
cv2.imshow('final', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
