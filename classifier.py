import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
from matplotlib import pyplot

class Classifier:
    def __init__(self):
        training_face_embadding_data = np.load('training_face_embaddings.npz')
        training_faces_embaddings = training_face_embadding_data['embaddings']
        training_faces_labels = training_face_embadding_data['labels']
        self._in_encoder = Normalizer(norm='l2')
        training_faces_embaddings = self._in_encoder.transform(training_faces_embaddings)
        self._out_encoder = LabelEncoder()
        self._out_encoder.fit(training_faces_labels)
        training_faces_labels = self._out_encoder.transform(training_faces_labels)
        self._model = SVC(kernel='linear', probability=True)
        self._model.fit(training_faces_embaddings, training_faces_labels)
        pass

    def classify(self, test_face, test_face_embadding):
        samples = np.expand_dims(test_face_embadding, axis=0)
        yhat_class = self._model.predict(samples)

        #TODO: see what happens if an unknown face is detected
        #TODO: no changes are required to be made for multiple faces in the image,
        # coz there will be no more than one face input to this function

        yhat_prob = self._model.predict_proba(samples)
        # get name
        class_index = yhat_class[0]
        print("Class name is: ",yhat_class)
        class_probability = yhat_prob[0,class_index] * 100
        predict_names = self._out_encoder.inverse_transform(yhat_class)
        print('Predicted: %s (%.3f)' % (predict_names[0], class_probability))
        return predict_names[0], class_probability
        # # plot for fun
        # pyplot.imshow(test_face)
        # title = '%s (%.3f)' % (predict_names[0], class_probability)
        # pyplot.title(title)
        # pyplot.show()

#
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
#
