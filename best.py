import cv2
import matplotlib.pyplot as plt
import numpy as np
import math

cap = cv2.VideoCapture("C:\\Users\\admin\\Videos\\VIRB\\VIRB_0001-2.mp4")
if not cap.isOpened():
    print("Could not open video file")
fig = plt.figure()
viewer = fig.add_subplot(111)
plt.ion()  # Turns interactive mode on (probably unnecessary)
fig.show()

while cap.isOpened():
    valid, frame = cap.read()
    frame = frame.astype("float32")
    frame = frame[:, :, 0] + frame[:, :, 1] + frame[:, :, 2]
    frame /= 3
    frame = frame.astype("uint8")
    if valid:
        sobel = cv2.Sobel(frame, cv2.CV_64F, 1, 1, ksize=3)
        abs_64 = np.absolute(sobel)
        sobel = np.uint8(abs_64)
        laplacian = cv2.Laplacian(frame, cv2.CV_64F)
        abs_64 = np.absolute(laplacian)
        laplacian = np.uint8(abs_64)
        frame = cv2.resize(frame, (500, 500))
        sobel = cv2.resize(sobel, (500, 500))
        laplacian = cv2.resize(laplacian, (500, 500))
        ris = np.concatenate((frame, 3 * sobel, 3 * laplacian), axis=1)
        cv2.imshow("best_of_best(sobel vs laplacian)", ris)
        cv2.waitKey(5)
    else:
        break

cap.release()
cv2.destroyAllWindows()
