import cv2
import mediapipe as mp
import time

########################
wCam, hCam = 640, 480
########################

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

pTime = 0

mpFaceDetection = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
faceDetection = mpFaceDetection.FaceDetection(0.75)

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceDetection.process(imgRGB)

    if results.detections:
        for id, detection in enumerate(results.detections):
            boundingBoxClass = detection.location_data.relative_bounding_box
            imageHeight, imageWidth, imageCenter = img.shape
            boundingBox = int(boundingBoxClass.xmin * imageWidth), int(boundingBoxClass.ymin * imageHeight),\
                          int(boundingBoxClass.width * imageWidth), int(boundingBoxClass.height * imageHeight)
            cv2.rectangle(img, (boundingBox[0], boundingBox[1]),
                          (boundingBox[0] + boundingBox[2], boundingBox[1] + boundingBox[3]), (255, 0, 255), 2)
            cv2.putText(img, f'{int(detection.score[0] * 100)}%', (boundingBox[0], boundingBox[1] - 5),
                        cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 255), 2)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 2)
    cv2.imshow("Image", img)

    if cv2.waitKey(1) & 0xff == ord('q'):
        break
