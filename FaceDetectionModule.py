import cv2
import mediapipe as mp
import time
import numpy as np


class FaceDetector:
    def __init__(self, min_detection_confidence: float = 0.7) -> None:
        self.minDetectionConfidence = min_detection_confidence
        self.mpFaceDetection = mp.solutions.face_detection
        self.mpDraw = mp.solutions.drawing_utils
        self.faceDetection = self.mpFaceDetection.FaceDetection(self.minDetectionConfidence)

    def findFaces(self, img: np.ndarray, draw: bool = True, color: list[int, int, int] = (255, 0, 255), thickness_box: int = 2,
                  font_scale: int = 1, thickness_text: int = 1) -> [np.ndarray, list[int, int, int, int]]:

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceDetection.process(imgRGB)
        boundingBoxes = []
        if self.results.detections:
            for id, detection in enumerate(self.results.detections):
                boundingBoxClass = detection.location_data.relative_bounding_box
                imageHeight, imageWidth, imageCenter = img.shape
                boundingBox = int(boundingBoxClass.xmin * imageWidth), int(boundingBoxClass.ymin * imageHeight),\
                              int(boundingBoxClass.width * imageWidth), int(boundingBoxClass.height * imageHeight)
                boundingBoxes.append([id, boundingBox, detection.score[0]])
                if draw:
                    cv2.rectangle(img, (boundingBox[0], boundingBox[1]),
                                  (boundingBox[0] + boundingBox[2], boundingBox[1] + boundingBox[3]),
                                  color, thickness_box)
                    cv2.putText(img, f'{int(detection.score[0] * 100)}%', (boundingBox[0], boundingBox[1] - 5),
                                cv2.FONT_HERSHEY_PLAIN, font_scale, color, thickness_text)
        return img, boundingBoxes


def main():
    ########################
    wCam, hCam = 640, 480
    ########################

    cap = cv2.VideoCapture(0)
    cap.set(3, wCam)
    cap.set(4, hCam)

    pTime = 0

    detector = FaceDetector()

    while True:
        success, img = cap.read()
        img, boundingBoxes = detector.findFaces(img)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 2)
        cv2.imshow("Image", img)

        if cv2.waitKey(1) & 0xff == ord('q'):
            break


if __name__ == "__main__":
    main()