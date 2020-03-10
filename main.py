import cv2
import numpy as np
from PIL import Image
from keras.models import load_model
from face_recognition_tf import FaceRecognitionTf
from classifier import Classifier

face_recognition_obj = FaceRecognitionTf()
classifier_obj = Classifier()

model = load_model('facenet_keras.h5')

capture_video = cv2.VideoCapture(0) # 0 to capture video from camera 0
frame_found, frame = capture_video.read()
while frame_found:
    array = np.asarray(frame)
    image = Image.fromarray(array)

    #TODO: program for multiple faces in the frame
    #TODO: only start processing if number of faces > 0
    #TODO: draw a box on the recognised face in this complete frame
    #TODO: maybe instead of showing live the frames we can save those frames in which we recognised someone
    face, (x1, y1, x2, y2) = face_recognition_obj.extract_face(image)
    if face.size == 0:
        print("Waiting for a face to appear!!!")
        continue
    embadding = face_recognition_obj.get_embedding(model, face)

    predicted_name, confidence = classifier_obj.classify(face, embadding)
    print("x1 y1 x2 y2", x1, y1, x2, y2)
    cv2.rectangle(frame, (x1, y1), (x2, y2), 20)
    cv2.imshow("{} confidence = {}".format(predicted_name, confidence), frame)
    cv2.waitKey(1)

    frame_found, frame = capture_video.read()
