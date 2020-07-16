import cv2 as cv

simile_cas=cv.CascadeClassifier('haarcascade_smile.xml')
face_cas=cv.CascadeClassifier('haarcascade_smile.xml')

def detect(gray, frame):
    faces = face_cas.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), ((x + w), (y + h)), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]
        smiles = smile_cas.detectMultiScale(roi_gray, 1.8, 20)

        for (sx, sy, sw, sh) in smiles:
            cv2.rectangle(roi_color, (sx, sy), ((sx + sw), (sy + sh)), (0, 0, 255), 2)
    return frame



web=cv.VideoCapture(0)

while True:

    ret,frame=web.read()
    gray=cv.cvtColor(frame,cv.COLOR_BGR2GRAY)

    smile_detect = detect(gray,frame)
    cv.imshow('video',smile_detect)
