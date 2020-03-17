import cv2
import numpy as np
import sys
from PIL import Image
from keras.models import load_model
from face_recognition_tf import FaceRecognitionTf
from classifier import Classifier
from datetime import datetime

face_recognition_obj = FaceRecognitionTf()
classifier_obj = Classifier()

model = load_model('facenet_keras.h5', compile=False)
total_faces_recognised = 0
capture_video = cv2.VideoCapture(0) # 0 to capture video from camera 0
frame_found, frame = capture_video.read()
while frame_found:
    array = np.asarray(frame)
    image = Image.fromarray(array)

    results, pixels = face_recognition_obj.extract_faces(image)
    no_of_faces = len(results)
    if no_of_faces == 0:
        # to keep printing in the same line
        sys.stdout.write("\rWaiting for a face to appear..."
                         +(" "*10)+"Total faces recognised = %d"%total_faces_recognised)
        sys.stdout.flush()
        frame_found, frame = capture_video.read()
        continue
    face_found = False
    sys.stdout.write('\r')
    # getting all the faces from right side
    while no_of_faces > 0:
        no_of_faces -= 1
        total_faces_recognised += 1
        face_found = True
        x1, y1, width, height = results[no_of_faces]['box']
        # bug fix
        x1, y1 = abs(x1), abs(y1)
        x2, y2 = x1 + width, y1 + height
        # extract the face
        face = pixels[y1:y2, x1:x2]
        # resize pixels to the model size
        image = Image.fromarray(face)
        image = image.resize(face_recognition_obj.image_size)
        face = np.asarray(image)

        embadding = face_recognition_obj.get_embedding(model, face)

        predicted_name, confidence = classifier_obj.classify(face, embadding)
        sys.stdout.write('Predicted: %s (%.3f), ' % (predicted_name, confidence)+(" "*10)
                         +"Total faces recognised = %d"%total_faces_recognised)
        sys.stdout.flush()

        cv2.rectangle(frame, (x1, y1), (x2, y2), (36, 255, 12))
        cv2.putText(frame, '%s (%.2f%%)'%(predicted_name, confidence), (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (36, 255, 12), 1)

    if face_found:
        file_name = "Recognised_faces/"+str(datetime.now()).replace(':', '-')+".jpg"
        #
        # Uncomment following line to save frames in the Recognised_faces folder
        #
        # cv2.imwrite(file_name, frame)
    cv2.imshow("frame", frame)
    if cv2.waitKey(1) == ord('a'):
        break

    frame_found, frame = capture_video.read()
