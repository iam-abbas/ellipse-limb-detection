
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
from math import atan2, degrees
from scipy.spatial import distance
import math
import numpy
import skvideo.io
import random
import csv

FILE_OUTPUT = 'output.avi'


def angle(p1, p2):
    return degrees(atan2(p1, p2))


def midp(x, y):
    return(int((x[0] + y[0])/2), int((x[1] + y[1])/2))


def focii(a, b):
    z = midp(a, b)
    z1 = midp(a, z)
    z2 = midp(z, b)
    return(z1, z2)


def dist(point1, point2):
    dist = math.hypot(point2[0] - point1[0], point2[1] - point1[1])
    return dist


protoFile = "models/pose_deploy_linevec_faster_4_stages.prototxt"
weightsFile = "models/pose_iter_160000.caffemodel"
nPoints = 15
POSE_PAIRS = [[0, 1], [2, 3], [3, 4], [5, 6], [6, 7],  [8, 9], [9, 10], [11, 12], [12, 13]]


net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

inWidth = 368
inHeight = 368

#cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture("video.avi")


fourcc = cv2.VideoWriter_fourcc(*'XVID') 
out = cv2.VideoWriter('ouput.avi', fourcc, 20.0, (960, 540))



j = 0
# while(cap.isOpened()):
while j < 100:
    j += 1
    ret, frame = cap.read()
    if cv2.waitKey(50) & 0xFF == ord('q'):
        break
    if ret == False:
        break

    #frame = cv2.imread("R5p88jk.jpg")

    frameCopy = np.copy(frame)
    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]
    threshold = 0.1

    if(j % 10 == 0):

        inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight),
                                        (0, 0, 0), swapRB=False)

        net.setInput(inpBlob)

        output = net.forward()
        H = output.shape[2]
        W = output.shape[3]

        # Empty list to store the detected keypoints
        points = []

        for i in range(nPoints):
            # confidence map of corresponding body's part.
            probMap = output[0, i, :, :]

            # Find global maxima of the probMap.
            minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

            # Scale the point to fit on the original image
            x = (frameWidth * point[0]) / W
            y = (frameHeight * point[1]) / H

            if prob > threshold:
                #cv2.circle(frameCopy, (int(x), int(y)), 8, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
                #cv2.putText(frameCopy, "{}".format(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, lineType=cv2.LINE_AA)
                #cv2.circle(frame, (int(x), int(y)), 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)

                # Add the point to the list if the probability is greater than the threshold
                points.append((int(x), int(y)))
            else:
                points.append(None)

        for pair in POSE_PAIRS:
            partA = pair[0]
            partB = pair[1]

            if points[partA] and points[partB]:

                #print(points[partA], points[partB])
                #print(focii(points[partA], points[partB]))
                #print((int((points[partA][0]+points[partB][0])/2), int((points[partA][1]+points[partB][1])/2)))

                cv2.ellipse(frame, (midp(points[partA], points[partB])), (int(dist(focii(points[partA], points[partB])[0], focii(points[partA], points[partB])[1])/2.5), int(dist(focii(points[partA], points[partB])[0], focii(points[partA], points[partB])[1]))), -1*angle(points[partA][0]-points[partB][0], points[partA][1]-points[partB][1]), 0, 360, (50, 0, 255), 1)
                #cv2.line(frame, points[partA], points[partB], (0, 255, 255), 3)
                #print(angle(points[partA][0]-points[partB][0], points[partA][1]-points[partB][1]))
                with open('main.csv', 'a+',  newline='') as writeFile:
                    writer = csv.writer(writeFile)
                    writer.writerows([["Fame-"+str(j), str(pair), (midp(points[partA], points[partB])), int(dist(focii(points[partA], points[partB])[0], focii(points[partA], points[partB])[1])), int(dist(focii(points[partA], points[partB])[0], focii(points[partA], points[partB])[1])/2.5), -1*angle(points[partA][0]-points[partB][0], points[partA][1]-points[partB][1])]])
                writeFile.close()
        # print(points)
        cv2.imwrite('images/'+str(j)+'.png', frame)
        #cv2.imshow("Cool", cv2.resize(cv2.cvtColor(frameCopy, cv2.COLOR_BGR2RGB), (960, 540)))
        #cv2.imwrite("images/"+str(random.randint(0, 1000))+".jpg", frame)

    cv2.imshow("Yes",  cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (960, 540)))
    out.write(cv2.resize(frame, (960, 540)))

out.release()
cap.release()
cv2.destroyAllWindows()
print("Done")
