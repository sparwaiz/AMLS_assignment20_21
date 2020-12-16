'''
ML Model For Predicting the Emotion from images.

The Model utilies dlib and keras.preprocessing module
to convert images to numpy arrays. Then Sklearn svm
model is trained to predict images
'''

import sys
from multiprocessing import Pool
from os import listdir, path

try:

    import numpy as np
    from cv2 import COLOR_BGR2GRAY, cvtColor
    from dlib import get_frontal_face_detector, shape_predictor
    from keras.preprocessing import image
    from sklearn import svm
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split
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

    _img = image.load_img(img_path,
                          target_size=None,
                          interpolation='bicubic')
    img = image.img_to_array(_img)

    feature, _ = run_dlib_shape(img)

    return feature, file_name

def shape_to_np(shape, dtype="int"):
    '''
    Utility Function to Convert a Shape
    to Numpy Array.

    shape: shape to be converted to the numpy array
    dtype: data type for numpy array
    '''
    coords = np.zeros((shape.num_parts, 2), dtype=dtype)

    for i in range(0, shape.num_parts):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    return coords


def rect_to_bb(rect):
    '''
    Converts dlib style rectangle to
    Open-CV Style Bounding Box

    rect: rectangle to be converted
    '''
    width = rect.right() - rect.left()
    height = rect.bottom() - rect.top()

    return width, height


def run_dlib_shape(img):
    '''
    Main Processing Function For the Image.

    The image is loaded to memory, preprocessed by
    first resizing it and converting to gray-scale.

    Then We utilise dlib shape predictor to figure out
    the 68 landmarks. The landmarks are then converted
    to a numpy array and returned along with the resized
    image

    img: image to be processed
    '''
    resized_img = img.astype('uint8')

    gray_scaled_img = cvtColor(resized_img, COLOR_BGR2GRAY)
    gray_scaled_img = gray_scaled_img.astype('uint8')

    rects = get_frontal_face_detector()(gray_scaled_img, 1)
    spred = path.dirname(path.realpath(__file__))
    spred = path.join(spred, 'shape_predictor_68_face_landmarks.dat')

    predictor = shape_predictor(spred)

    num_faces = len(rects)

    if not num_faces:
        return None, resized_img

    face_areas = np.zeros((1, num_faces))
    face_shapes = np.zeros((136, num_faces), dtype=np.int64)

    for i, rect in enumerate(rects):
        temp_shape = predictor(gray_scaled_img, rect)
        temp_shape = shape_to_np(temp_shape)

        width, height = rect_to_bb(rect)

        face_shapes[:, i] = np.reshape(temp_shape, [136])
        face_areas[0, i] = width * height

    dlibout = np.reshape(np.transpose(face_shapes[:, np.argmax(face_areas)]),
                         [68, 2])

    return dlibout, resized_img


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
                tqdm(extractor.imap(convert_to_feature, images),
                     total=total))

        images = filter(lambda image: not isinstance(image[0], type(None)),
                        images)

        __labels = self.__get_labels(self.label_file)

        features, labels = zip(*images)

        labels = map(__labels.get, labels)

        return np.array(features), (np.fromiter(labels, dtype=np.int) + 1) / 2

    def __split_data(self):
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

        training_images = tr_x.reshape((len(tr_x), len(tr_x[0]) * 2))
        training_labels = list(zip(*tr_y))[0]

        validation_images = vl_x.reshape((len(vl_x), len(vl_x[0]) * 2))
        validation_labels = list(zip(*vl_y))[0]

        self.testing_images = te_x.reshape((len(te_x), len(te_x[0]) * 2))
        self.testing_labels = list(zip(*te_y))[0]

        return training_images, training_labels, validation_images, validation_labels

    def __train_model(self):
        '''
        private method to train the ML model
        on processed dataset
        '''
        if self.model is not None:
            return None, None

        training_images, training_labels, validation_images, validation_labels = self.__split_data()

        self.model = svm.SVC(kernel='poly', degree=3, C=1.0)
        self.model.fit(training_images, training_labels)

        return validation_images, validation_labels

    def validate(self):
        '''
        Validate Method to test model on validation data
        '''
        validation_images, validation_labels = self.__train_model()
        pred = self.model.predict(validation_images)

        print(
            f'Accuracy of Model on validation set is {colored(accuracy_score(validation_labels, pred), "green")}'
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
        extra_images = data_x.reshape((nsamples, nx*ny))
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
            line.split('\t')[0]: int(line.split('\t')[3])
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
    model = Model('..', 'celeba')
    model.validate()
    model.predict()
    model.predict(extra=True)
