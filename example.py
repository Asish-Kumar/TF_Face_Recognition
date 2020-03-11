import cv2
capture = cv2.VideoCapture(0)
image_found, frame = capture.read()
i = 0
while image_found:
    cv2.imshow("frame", frame)
    image_found, frame = capture.read()
    cv2.imwrite("Dataset/train/Surander_bhaiya/image%d.jpg"%i, frame)
    if cv2.waitKey(1) == ord('a'):
        break
    i += 1
