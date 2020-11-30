'''
ML Model For Predicting the Gender from images.

The Model utilies dlib and keras.preprocessing module
to convert images to numpy arrays. Then Sklearn svm
model is trained to predict images
'''

import sys
from multiprocessing import Pool
from os import listdir, path

try:

    import numpy as np
    import cv2
    from sklearn.svm import SVC
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split
    from tqdm import tqdm

except ImportError:
    print('Required Modules Not Found')
    sys.exit(1)

def convert_to_feature(img_path):
    '''
    Function To Convert Image Path
    to it's Features.

    return value: features for the image,
    and file_name without extension

    img_path: file path
    '''
    file_name = img_path.split('.')[-2].split('/')[-1]
    img = img_path

    return img, file_name

class Model:
    '''
    Main ML Model Class that acts
    as the abstraction layer to
    run the ML model
    '''
    def __init__(self, project_root, dataset):
        self.__dataset = dataset
        self.__project_root = project_root

        self.model = None
        self.label_file = None

        self.testing_images = None
        self.testing_labels = None

    def extract_features(self):
        '''
        Extract Features and Labels From Images
        '''
        print('Extracting Features')
        images, total = self.__get_images()

        with Pool() as extractor:
            images = list(
                tqdm(extractor.imap(convert_to_feature, images),
                     total=total))

        images = filter(lambda image: not isinstance(image[0], type(None)),
                        images)

        __labels = self.__get_labels(self.label_file)

        features, labels = zip(*images)

        labels = map(__labels.get, labels)

        labels = np.fromiter(labels, dtype=np.int)
        features = np.array(features)

        return features, labels

    def __split_data(self):
        data_x, data_y = self.extract_features()
        data_y = np.array([data_y, -(data_y - 1)]).T

        tr_x, te_x, tr_y, te_y = train_test_split(data_x,
                                                  data_y,
                                                  test_size=0.25,
                                                  random_state=0)

        nsamples, nx, ny = tr_x.shape
        training_images = tr_x.reshape((nsamples, nx*ny))
        training_labels = list(zip(*tr_y))[0]


        nsamples, nx, ny = te_x.shape
        self.testing_images = te_x.reshape((nsamples, nx*ny))
        self.testing_labels = list(zip(*te_y))[0]

        return training_images, training_labels

    def __train_model(self):
        training_images, training_labels = self.__split_data()

        self.model = SVC()
        self.model.fit(training_images, training_labels)

    def predict(self):
        '''
        Predict Methord to Be Called that starts predicting
        '''
        self.__train_model()
        pred = self.model.predict(self.testing_images)

        print(
            f'Accuracy of Model is {accuracy_score(self.testing_labels, pred)}'
        )

        return pred

    @staticmethod
    def __get_labels(label_file):
        with open(label_file, 'r') as labels_file:
            lines = labels_file.readlines()

        labels = {
            line.split('\t')[0]: int(line.split('\t')[1])
            for line in lines[1:]
        }

        return labels



    def __get_images(self):
        __dir = path.join(self.__project_root, 'Datasets')
        __dir = path.join(__dir, self.__dataset)

        self.label_file = path.join(__dir, 'labels.csv')
        __dir = path.join(__dir, 'img')

        images = listdir(__dir)

        return map(lambda img: path.join(__dir, img), images), len(images)


if __name__ == '__main__':
    model = Model('..', 'cartoon_set')
    model.predict()
