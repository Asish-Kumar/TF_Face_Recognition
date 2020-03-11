import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC

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

        #TODO: see what happens if an unknown face is detected --> it is left unrecognised
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
