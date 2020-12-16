'''
ML Model For Predicting the Face Shape from images.

The Model utilies opencv to preprocess anr remove noise
from images and convert images to numpy arrays. Then 
Sklearn svm model is trained to predict images
'''

import sys
from multiprocessing import Pool
from os import listdir, path

try:

    import cv2
    import numpy as np
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split
    from sklearn.svm import SVC
    from termcolor import colored
    from tqdm import tqdm

except ImportError:
    print('Dependencies not satisfied')
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
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.medianBlur(img, 5)
    img = cv2.resize(img, (50, 50))

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

    def extract_features(self, extra=False):
        '''
        Extract Features and Labels From Images
        '''
        if extra:
            print('Extracting Features From Extra Dataset')
        else:
            print('Extracting Features From Original Dataset')

        images, total = self.__get_images(extra)

        with Pool() as extractor:
            images = list(
                tqdm(extractor.imap(convert_to_feature, images), total=total))

        images = filter(lambda image: not isinstance(image[0], type(None)),
                        images)

        __labels = self.__get_labels(self.label_file)

        features, labels = zip(*images)

        labels = map(__labels.get, labels)

        labels = np.fromiter(labels, dtype=np.int)
        features = np.array(features)

        return features, labels

    def __split_data(self):
        '''
        private method to split the data into 3
        parts:
            - training (for training the model)
            - validation (To validate the model i.e. avoid overfitting)
            - testing to test the model
        '''
        data_x, data_y = self.extract_features()
        data_y = np.array([data_y, -(data_y - 1)]).T

        tr_x, te_x, tr_y, te_y = train_test_split(data_x,
                                                  data_y,
                                                  test_size=0.25,
                                                  random_state=0)

        tr_x, vl_x, tr_y, vl_y = train_test_split(tr_x,
                                                  tr_y,
                                                  test_size=0.25,
                                                  random_state=0)

        nsamples, nx, ny = tr_x.shape
        training_images = tr_x.reshape((nsamples, nx * ny))
        training_labels = list(zip(*tr_y))[0]

        nsamples, nx, ny = vl_x.shape
        validation_images = vl_x.reshape((nsamples, nx * ny))
        validation_labels = list(zip(*vl_y))[0]

        nsamples, nx, ny = te_x.shape
        self.testing_images = te_x.reshape((nsamples, nx * ny))
        self.testing_labels = list(zip(*te_y))[0]

        return training_images, training_labels, validation_images, validation_labels

    def __train_model(self):
        '''
        private method to train the ML model
        on processed dataset
        '''
        if self.model is not None:
            return None, None

        training_images, training_labels, validation_images, validation_labels = self.__split_data(
        )

        self.model = SVC()
        self.model.fit(training_images, training_labels)

        return validation_images, validation_labels

    def validate(self):
        '''
        Validate Method to be Called that tests out the
        validation data
        '''
        validation_images, validation_labels = self.__train_model()

        pred = self.model.predict(validation_images)

        print(
            f'Accuracy of Model on Validation Set {colored(accuracy_score(validation_labels, pred), "green")}'
        )

    def predict(self, extra=False):
        '''
        Predict Method to Be called that starts predicting
        on the test data
        '''
        self.__train_model()

        if not extra:
            pred = self.model.predict(self.testing_images)

            print(
                f'Accuracy of Model on Testing Set {colored(accuracy_score(self.testing_labels, pred), "green")}'
            )

            return pred

        data_x, data_y = self.extract_features(extra)
        data_y = np.array([data_y, -(data_y - 1)]).T

        nsamples, nx, ny = data_x.shape
        extra_images = data_x.reshape((nsamples, nx * ny))
        extra_labels = list(zip(*data_y))[0]

        pred = self.model.predict(extra_images)

        print(
            f'Accuracy of Model on Extra Testing Set {colored(accuracy_score(extra_labels, pred), "green")}'
        )

        return pred

    @staticmethod
    def __get_labels(label_file):
        '''
        label_file: csv file for the dataset

        private class function to read the csv
        file and return labels for training the
        model
        '''
        with open(label_file, 'r') as labels_file:
            lines = labels_file.readlines()

        labels = {
            line.split('\t')[0]: int(line.split('\t')[2])
            for line in lines[1:]
        }

        return labels

    def __get_images(self, extra=False):
        '''
        extra (boolean): use the extra dataset

        private class function to create path to dataset
        and return list og images in the directory
        '''
        __dir = path.join(self.__project_root, 'Datasets')
        __dir = path.join(__dir, self.__dataset)

        if extra:
            __dir = __dir + "_test"

        self.label_file = path.join(__dir, 'labels.csv')
        __dir = path.join(__dir, 'img')

        images = listdir(__dir)

        return map(lambda img: path.join(__dir, img), images), len(images)


if __name__ == '__main__':
    model = Model('..', 'cartoon_set')
    model.validate()
    model.predict()
    model.predict(extra=True)
