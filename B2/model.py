'''
ML Model For Predicting the Eye Color from Cartoon images.

The Model utilies opencv to find the pupil in the image and
convert it to a numpy array,Then Sklearn svm model is trained
to predict images
'''

import sys
from multiprocessing import Pool
from os import listdir, path

try:

    import cv2
    import numpy as np
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split, learning_curve
    from sklearn.svm import SVC
    from termcolor import colored
    from tqdm import tqdm
    import matplotlib.pyplot as plt

except ImportError:
    print('Dependencies not satisfied')
    sys.exit(1)


def plot_learning_curve(estimator, title, X, y, axes=None, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate 3 plots: the test and training learning curve, the training
    samples vs fit times curve, the fit times vs score curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    axes : array of 3 axes, optional (default=None)
        Axes to use for plotting the curves.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

          - None, to use the default 5-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, optional (default=None)
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like, shape (n_ticks,), dtype float or int
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the dtype is float, it is regarded as a
        fraction of the maximum size of the training set (that is determined
        by the selected validation method), i.e. it has to be within (0, 1].
        Otherwise it is interpreted as absolute sizes of the training sets.
        Note that for classification the number of samples usually have to
        be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(20, 5))

    axes[0].set_title(title)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = \
        learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                       train_sizes=train_sizes,
                       return_times=True)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    print(train_scores_mean)

    # Plot learning curve
    axes[0].grid()
    axes[0].fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    axes[0].fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
    axes[0].plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
    axes[0].plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
    axes[0].legend(loc="best")

    # Plot n_samples vs fit_times
    axes[1].grid()
    axes[1].plot(train_sizes, fit_times_mean, 'o-')
    axes[1].fill_between(train_sizes, fit_times_mean - fit_times_std,
                         fit_times_mean + fit_times_std, alpha=0.1)
    axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("fit_times")
    axes[1].set_title("Scalability of the model")

    # Plot fit_time vs score
    axes[2].grid()
    axes[2].plot(fit_times_mean, test_scores_mean, 'o-')
    axes[2].fill_between(fit_times_mean, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1)
    axes[2].set_xlabel("fit_times")
    axes[2].set_ylabel("Score")
    axes[2].set_title("Performance of the model")

    return plt

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

    gray_scaled = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_scaled = cv2.medianBlur(gray_scaled, 5)

    circles = cv2.HoughCircles(gray_scaled,
                               cv2.HOUGH_GRADIENT,
                               0.5,
                               img.shape[0],
                               param1=440,
                               param2=14,
                               minRadius=11,
                               maxRadius=12)

    if circles is not None:
        eye_coords = circles[0][-1]

        x_coord = int(eye_coords[0])
        y_coord = int(eye_coords[1])
        radius = int(eye_coords[2])

        eye = img[y_coord:y_coord + radius, x_coord:x_coord + radius]
        eye = cv2.cvtColor(eye, cv2.COLOR_BGR2HSV)
        eye = cv2.resize(eye, (50, 50))

        return eye, file_name

    return None, file_name


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

        self.X = None
        self.y = None

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

        with Pool(4) as extractor:
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

        nsamples, nx, ny, nz = tr_x.shape
        training_images = tr_x.reshape((nsamples, nx * ny * nz))
        training_labels = list(zip(*tr_y))[0]

        nsamples, nx, ny, nz = vl_x.shape
        validation_images = vl_x.reshape((nsamples, nx * ny * nz))
        validation_labels = list(zip(*vl_y))[0]

        nsamples, nx, ny, nz = te_x.shape
        self.testing_images = te_x.reshape((nsamples, nx * ny * nz))
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

        self.X = training_images
        self.y = training_labels

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

        nsamples, nx, ny, nz = data_x.shape
        extra_images = data_x.reshape((nsamples, nx * ny * nz))
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
            line.split('\t')[0]: int(line.split('\t')[1])
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
    plot_learning_curve(SVC(verbose=True), r"Learning Curves (SVM, RBF Kernel)", model.X, model.y)
    plt.show()
