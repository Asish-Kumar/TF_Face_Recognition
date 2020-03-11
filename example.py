import cv2
from datetime import datetime
campture = cv2.VideoCapture(0)
image_found, frame = campture.read()
i = 0
start = datetime.now()
print("start is:", start)
while image_found:
    cv2.imshow("frame", frame)
    image_found, frame = campture.read()

    if cv2.waitKey(1) == ord('a'):
        stop = datetime.now()
        stop_time = stop.time()
        start_time = start.time()
        t = stop - start
        print(start_time, stop_time, i, t)
        break
    i+=1



