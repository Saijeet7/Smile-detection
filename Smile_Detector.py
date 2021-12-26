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
    #Show the current frame
    cv2.imshow('Smile detection', frame)
    #Display
    cv2.waitKey(1)

#Clean up
webcam.release()
cv2.destroyAllWindows()