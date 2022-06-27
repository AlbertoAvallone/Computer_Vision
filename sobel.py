import cv2
import matplotlib.pyplot as plt
import numpy as np


cap = cv2.VideoCapture("C:\\Users\\admin\\Videos\\VIRB\\VIRB_0001-2.mp4")

# img = cv2.imread("test_data\\first_example.png")

if not cap.isOpened():
    print("Could not open video file")

#cv2.namedWindow("original", cv2.WINDOW_NORMAL)
#cv2.namedWindow("edge", cv2.WINDOW_NORMAL)

fig = plt.figure()
viewer = fig.add_subplot(111)
plt.ion()  # Turns interactive mode on (probably unnecessary)
fig.show()

#Parameters

#ksizes = [1, 3, 5, 7]
ksizes = [1, 3, 5]
dim = [1, 2]
#names = ["ksizes_1", "ksizes_3", "ksizes_5", "ksizes_7"]

while cap.isOpened():
    valid, frame = cap.read()
    if valid:
        edges = list()
        for d in dim:
            for ks in ksizes:
                edge = cv2.Sobel(frame, cv2.CV_64F, d, 0, ksize=ks)
                abs_64 = np.absolute(edge)
                edge = np.uint8(abs_64)
                edge = cv2.resize(edge, (250, 250))
                edges.append(edge)
            for ks in ksizes:
                edge = cv2.Sobel(frame, cv2.CV_64F, 0, d, ksize=ks)
                abs_64 = np.absolute(edge)
                edge = np.uint8(abs_64)
                edge = cv2.resize(edge, (250, 250))
                edges.append(edge)
            for ks in ksizes:
                edge = cv2.Sobel(frame, cv2.CV_64F, d, d, ksize=ks)
                abs_64 = np.absolute(edge)
                edge = np.uint8(abs_64)
                edge = cv2.resize(edge, (250, 250))
                edges.append(edge)

        for row in range(len(dim)):
            new_im = edges[row * len(ksizes)]
            for col in range(1, len(ksizes)):
                new_im = np.concatenate((new_im, edges[col + row*len(ksizes)]), axis=1)
            if row == 0:
                axis_x = new_im
            else:
                axis_x = np.concatenate((axis_x, new_im), axis=0)

        for row in range(len(dim)):
            new_im = edges[row * len(ksizes)]
            for col in range(1, len(ksizes)):
                new_im = np.concatenate((new_im, edges[col + row*len(ksizes)]), axis=1)
            if row == 0:
                axis_y = new_im
            else:
                axis_y = np.concatenate((axis_y, new_im), axis=0)

        for row in range(len(dim)):
            new_im = edges[row * len(ksizes)]
            for col in range(1, len(ksizes)):
                new_im = np.concatenate((new_im, edges[col + row*len(ksizes)]), axis=1)
            if row == 0:
                axis_xy = new_im
            else:
                axis_xy = np.concatenate((axis_xy, new_im), axis=0)


        #cv2.imshow("sobel_x", axis_x)
        cv2.imshow("sobel_y", axis_y)
        cv2.imshow("sobel_xy", axis_xy)
        cv2.waitKey(5)
    else:
        break
cap.release()
cv2.destroyAllWindows()