import cv2
from numba import jit
import face_recognition
from threading import Thread


def get_face_locations(pic):
    locations = face_recognition.face_locations(pic)
    return locations


def get_face_encoding(pic, face_location):
    person_face_encoding = face_recognition.face_encodings(pic, face_location)
    return person_face_encoding



capture_video = cv2.VideoCapture(0) # 0 to capture video from camera 0
frame_found, frame = capture_video.read()
while frame_found:
    face_loc = get_face_locations(frame)
    num_faces = len(face_loc)
    # filtration : only process those frames which have atleast one face present
    if num_faces > 0:
        # put this frame for further processing
        for (top, right, bottom, left) in face_loc:
            cv2.rectangle(frame, (left, top), (right, bottom), 20)
        cv2.imshow("Window", frame)
        cv2.waitKey(1)
        #encoding = get_face_encoding(frame, face_loc)
        frame_found, frame = capture_video.read()






"""
ALGORITHM:
1. Capture Video 
2. Read frames from it
3. pass these frames to face_recognition to detect if a face is present
4. save those frames which have atleast one face present in it

"""

