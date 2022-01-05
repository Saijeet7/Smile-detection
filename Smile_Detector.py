import cv2

#Face Classifier
face_detector = cv2.CascadeClassifier('haarcascade_frontface.xml')
smile_detector = cv2.CascadeClassifier('haarcascade_smile.xml')

#Grab webcam feed
webcam = cv2.VideoCapture(0)

while True:
    #Read current frame from webcam video stream
    successful_frame_read, frame = webcam.read()

    #If there's an error, abort
    if not successful_frame_read:
        break

    #Change to grayScale
    frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #Detect faces first
    faces = face_detector.detectMultiScale(frame_grayscale)
    smiles = smile_detector.detectMultiScale(frame_grayscale, scaleFactor=1.7, minNeighbors=20)
    #Face detection rectangle
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (100, 200, 50), 4)
    for (x, y, w, h) in smiles:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (50, 50, 200), 4)

    #Show the current frame
    cv2.imshow('Smile detection', frame)
    #Display

    #Dont autoclose 
    key = cv2.waitKey(1)
    
    #Press Q for quit
    if key ==81 or key==113:
        break

#Clean up
webcam.release()
cv2.destroyAllWindows()