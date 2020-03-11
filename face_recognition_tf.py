from os import listdir
from os.path import isdir
from PIL import Image
import numpy as np
import sys
from mtcnn.mtcnn import MTCNN

"""
ISSUES:
1. During testing a single photo will have multiple faces , our extract_faces function only extracts one face

"""

np.set_printoptions(threshold=sys.maxsize)


class FaceRecognitionTf:
    def __init__(self):
        self.image_size = (160, 160)
        pass

    def extract_faces(self, image):
        """Extract all the faces from a given photograph. It returns list of dict (one dict for one face)
        and image pixel matrix."""
        # convert to RGB, if needed
        image = image.convert('RGB')
        # convert to array
        pixels = np.asarray(image)
        # create the detector, using default weights
        detector = MTCNN()
        # detect faces in the image
        results = detector.detect_faces(pixels)  # returns list of dict (one dict for one face)
        print("Detected face result is :", results)
        return results, pixels

    def load_faces(self, directory):
        """Load images and extract faces for all images in a directory.
        This function is only supposed to be used for training the model therefore it considers only one face in
        one image."""
        faces = list()
        # enumerate files
        for filename in listdir(directory):
            # path
            path = directory + filename
            # load image from file
            image = Image.open(path)
            # get face
            results, pixels = self.extract_faces(image)
            # continue if no faces were found
            if len(results) == 0:
                continue
            # get the coordinates for first face in the image from left side
            x1, y1, width, height = results[0]['box']
            # bug fix
            x1, y1 = abs(x1), abs(y1)
            x2, y2 = x1 + width, y1 + height
            # extract the face
            face = pixels[y1:y2, x1:x2]
            # resize pixels to the model size
            image = Image.fromarray(face)
            image = image.resize(self.image_size)
            face = np.asarray(image)
            # store
            faces.append(face)
        return faces

    def load_dataset(self, directory):
        """Load a dataset that contains one subdir for each class that in turn contains images."""
        X, y = list(), list()
        # X will contain faces and Y will contain labels
        # enumerate folders, on per class
        for subdir in listdir(directory):
            # path
            path = directory + subdir + '/'
            # skip any files that might be in the dir
            if not isdir(path):
                continue
            # load all faces in the subdirectory
            faces = self.load_faces(path)
            # create labels
            labels = [subdir for _ in range(len(faces))]
            # summarize progress
            print('>loaded %d examples for class: %s' % (len(faces), subdir))
            # store
            X.extend(faces)
            y.extend(labels)
        return np.asarray(X), np.asarray(y)

    def get_embedding(self, model, face_pixels):
        """get the face embedding for one face"""
        # scale pixel values
        face_pixels = face_pixels.astype('float32')
        # standardize pixel values across channels (global)
        mean, std = face_pixels.mean(), face_pixels.std()
        face_pixels = (face_pixels - mean) / std
        # transform face into one sample
        samples = np.expand_dims(face_pixels, axis=0)
        # make prediction to get embedding
        yhat = model.predict(samples)
        return yhat[0]


#
# # develop a classifier for the 5 Celebrity Faces Dataset
# from random import choice
# from numpy import load
# from numpy import expand_dims
# from sklearn.preprocessing import LabelEncoder
# from sklearn.preprocessing import Normalizer
# from sklearn.svm import SVC
# from matplotlib import pyplot
# # load faces
# data = load('5-celebrity-faces-dataset.npz')
# testX_faces = data['arr_2']
# # load face embeddings
# data = load('5-celebrity-faces-embeddings.npz')
# trainX, trainy, testX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
# # normalize input vectors
# in_encoder = Normalizer(norm='l2')
# trainX = in_encoder.transform(trainX)
# testX = in_encoder.transform(testX)
# # label encode targets
# out_encoder = LabelEncoder()
# out_encoder.fit(trainy)
# trainy = out_encoder.transform(trainy)
# testy = out_encoder.transform(testy)
# # fit model
# model = SVC(kernel='linear', probability=True)
# model.fit(trainX, trainy)
# # test model on a random example from the test dataset
# selection = choice([i for i in range(testX.shape[0])])
# random_face_pixels = testX_faces[selection]
# random_face_emb = testX[selection]
# random_face_class = testy[selection]
# random_face_name = out_encoder.inverse_transform([random_face_class])
# # prediction for the face
# samples = expand_dims(random_face_emb, axis=0)
# yhat_class = model.predict(samples)
# yhat_prob = model.predict_proba(samples)
# # get name
# class_index = yhat_class[0]
# class_probability = yhat_prob[0,class_index] * 100
# predict_names = out_encoder.inverse_transform(yhat_class)
# print('Predicted: %s (%.3f)' % (predict_names[0], class_probability))
# print('Expected: %s' % random_face_name[0])
# # plot for fun
# pyplot.imshow(random_face_pixels)
# title = '%s (%.3f)' % (predict_names[0], class_probability)
# pyplot.title(title)
# pyplot.show()
