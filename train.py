"""this will not remove any duplicate face data,
 so inorder to add new suspect to the dataset,
 first remove the folders containing already added suspects and then run this file,
 and inorder to add new faces for already added suspect,
 first remove previously added photos of the suspect and then run this file,
 while the training folders contain only new photos"""
from os import path
import numpy as np
from keras.models import load_model
from face_recognition_tf import FaceRecognitionTf

face_recognition_obj = FaceRecognitionTf()

"""
PENDING:
1. Calculate performance gain by using GPU over CPU
"""

model = load_model('facenet_keras.h5')
print('Loaded Model')
# load train dataset
faces, labels = face_recognition_obj.load_dataset('Dataset/train/')
print(faces.shape, labels.shape)

embaddings = list()
for face_pixels in faces:
    embedding = face_recognition_obj.get_embedding(model, face_pixels)
    embaddings.append(embedding)
embaddings_array = np.asarray(embaddings)
print("New embadding shape:", embaddings_array.shape)

# check if any previous face embaddings data exists
if path.exists('training_face_embaddings.npz'):
    print('Previous face embadding data found...')
    prev_data = np.load('training_face_embaddings.npz')
    print(prev_data['embaddings'].shape, prev_data['labels'].shape)
    # append previous dataset with current
    embaddings_array = np.append(embaddings_array, prev_data['embaddings'], axis=0)
    labels = np.append(labels, prev_data['labels'], axis=0)

# check if any previous dataset exists
if path.exists('training_dataset.npz'):
    print("Previous training dataset found...")
    prev_data = np.load('training_dataset.npz')
    print(prev_data['faces'].shape, prev_data['labels'].shape)
    # append previous dataset with current
    faces = np.append(faces, prev_data['faces'], axis=0)
    # following line is not required as the same has been done previously above
    # labels = np.append(labels, prev_data['labels'], axis=0)

# save arrays to one file in compressed format
np.savez_compressed('training_dataset.npz', faces=faces, labels=labels)
np.savez_compressed('training_face_embaddings.npz', embaddings=embaddings_array, labels=labels)
print("="*30)
print(faces.shape, labels.shape)
