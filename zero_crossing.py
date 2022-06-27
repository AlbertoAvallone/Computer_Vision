import cv2
import matplotlib.pyplot as plt
import numpy as np

cap = cv2.VideoCapture("C:\\Users\\admin\\Videos\\VIRB\\VIRB_0001-2.mp4")

# img = cv2.imread("test_data\\first_example.png")

if not cap.isOpened():
    print("Could not open video file")

fig = plt.figure()
viewer = fig.add_subplot(111)
plt.ion()  # Turns interactive mode on (probably unnecessary)
fig.show()

format = [cv2.CV_32F, cv2.CV_64F, cv2.CV_16S, cv2.CV_16U]

while cap.isOpened():
    valid, frame = cap.read()
    if valid:
        edges = list()
        for sgn in format:
            ris = cv2.Laplacian(frame, sgn)
            ris = ris / ris.max()
            ris = cv2.resize(ris, (500, 500))
            edges.append(ris)

        axis_x = 2
        axis_y = 2

        for y in range(axis_y):
            for x in range(axis_x):
                if x == 0:
                    tmp = edges[0]
                else:
                    tmp = np.concatenate((tmp, edges[axis_x * y + x]), axis=1)
            if y == 0:
                tot = tmp
            else:
                tot = np.concatenate((tot, tmp), axis=0)

        frame = cv2.resize(frame, (500, 500))
        cv2.imshow("frame", frame)
        cv2.imshow("result", 3 * tot)
        cv2.waitKey(10)
    else:
        break