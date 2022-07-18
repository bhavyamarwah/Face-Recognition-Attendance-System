import cv2     #opencv= open source computer vision library for face detection
import numpy as np
import face_recognition
import os         #for reading
from datetime import datetime   #for attendance marking date and time


path = 'images'
images = []
personNames = []
myList = os.listdir(path)          #used to read the path specified in it
print(myList)
for cu_img in myList:
    current_Img = cv2.imread(f'{path}/{cu_img}')           #cv2 is used to read the images (path,flag)
    images.append(current_Img)
    personNames.append(os.path.splitext(cu_img)[0])
print(personNames)


def faceEncodings(images):                            #this function is made as dlib encodes the face into 128 encodings,
    encodeList = []                                   #from which we can uniquely identify a face.
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)       #from one colour space to another
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList


def attendance(name):
    with open('Attendance.csv', 'r+') as f:     #read append mode (f object)
        myDataList = f.readlines()               #has all the data
        nameList = []                           #empty list created
        for line in myDataList:
            entry = line.split(',')                  #comma separated value
            nameList.append(entry[0])                #append 0th element
        if name not in nameList:
            time_now = datetime.now()                 #new entry with current date and time
            tStr = time_now.strftime('%H:%M:%S')       #string for time
            dStr = time_now.strftime('%d/%m/%Y')        #string for date
            f.writelines(f'\n{name},{tStr},{dStr}')


encodeListKnown = faceEncodings(images)
print('All Encodings Complete!!!')

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()          #returns 2 things: ret variable and camera's frame
    faces = cv2.resize(frame, (0, 0), None, 0.25, 0.25)
    faces = cv2.cvtColor(faces, cv2.COLOR_BGR2RGB)

    facesCurrentFrame = face_recognition.face_locations(faces)            #finds faces in the current frame
    encodesCurrentFrame = face_recognition.face_encodings(faces, facesCurrentFrame)     #encodes the current frame

#now we have to see ki jo face hume dikh rha hai aur jo current frame mai aa rha hai wo match ho rha hai ya nhi

    for encodeFace, faceLoc in zip(encodesCurrentFrame, facesCurrentFrame):   #zip is used for passing more than one parameters together
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)  #to check face matching
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)  #for face distance: less dist= matches, more dist= doesnt match
        # print(faceDis)
        matchIndex = np.argmin(faceDis)       #takes out the minimum distance's index value

        if matches[matchIndex]:    #checks jo humara camera se image aa rha hai is same as stored in directory
            name = personNames[matchIndex].upper()
            # print(name)
            y1, x2, y2, x1 = faceLoc                 #dimension according to face_rec library
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4         #since 0.25 kia tha toh resize to original
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)        #dimension and colour code of rectangle
            cv2.rectangle(frame, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)   #rectangle for name above frame(-35 pixels)
            cv2.putText(frame, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            attendance(name)

    cv2.imshow('Webcam', frame)
    if cv2.waitKey(1) == 13:  #checks in every 1 ms ki value == 13 or not (13= enter key)
        break

cap.release()
cv2.destroyAllWindows() #destroy all windows