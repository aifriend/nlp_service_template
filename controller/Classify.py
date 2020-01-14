import warnings

from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from tensorflow_core.python.keras.utils.np_utils import to_categorical

from classifier.NNetwork import NNetwork
from classifier.NaiveBayes import NaiveBayes
from classifier.Voting import Voting
from controller.ClassFile import ClassFile

warnings.filterwarnings('ignore')  # "error", "ignore", "always", "default", "module" or "once"


class Classify:
    NAIVE_BAYES_MULTI = "NAIVE_BAYES_MULTI"
    VOTING = "VOTING"
    CNN_NETWORK = "CNN_NETWORK"

    def __init__(self, conf):
        self.conf = conf
        self.from_file = ClassFile()

        self._naive_bayes = NaiveBayes(conf)
        self._cnn_network = NNetwork(conf)
        self._voting = Voting(conf)

    def get_category(self, gram_path):
        return self.from_file.get_containing_dir_name(gram_path)

    @staticmethod
    def encode_categories(y):
        # encode class values as integers
        encoder = LabelEncoder()
        encoded_Y = encoder.fit_transform(y)
        # convert integers to dummy variables (i.e. one hot encoded)
        dummy_y = to_categorical(encoded_Y).astype(int)
        # print(dummy_y)
        return dummy_y

    @staticmethod
    def show_metrics(y_test, y_predicted, stats=None, show=False):
        print('Accuracy:', accuracy_score(y_test, y_predicted))
        print(metrics.classification_report(y_test, y_predicted))
        stats.info = metrics.classification_report(y_test, y_predicted)
        return stats

    def launch_naive_bayes_complement(self, X_train=None, y_train=None, X_test=None, train=False):
        print('---< Naive-Bayes Complement >---')
        result = ''
        if train and X_train is not None and y_train is not None:
            self._naive_bayes.train(X_train, y_train, 'complement')
        elif not train and X_test is not None:
            result = self._naive_bayes.predict(X_test, 'complement')

        return result

    def launch_naive_bayes_multinomial(self, X_train=None, y_train=None, X_test=None, train=False):
        print('---< Naive-Bayes Multinomial >---')
        result = ''
        if train and X_train is not None and y_train is not None:
            self._naive_bayes.train(X_train, y_train, 'multinomial')
        elif not train and X_test is not None:
            result = self._naive_bayes.predict(X_test, 'multinomial')

        return result

    def launch_cnn_network(self, training_set=None, validation_set=None, prediction_set=None, train=False):
        print('---< Nn Network >---')
        result = ''
        if train and training_set is not None and validation_set is not None:
            self._cnn_network.train(training_set, validation_set)
        elif not train and prediction_set is not None:
            result = self._cnn_network.predict(prediction_set)

        return result

    def launch_voting_classifier(self, X_train=None, y_train=None, X_test=None, train=False):
        print('---< Voting Classifier >---')
        result = ''
        clf_nb = self._naive_bayes.initialize(subtype='complement')
        class_list = [clf_nb]
        if train and X_train is not None and y_train is not None:
            self._voting.train(X_train, y_train, class_list)
        elif not train and X_test is not None:
            result = self._voting.predict(X_test, class_list)

        return result
