import os

from keras_preprocessing.image import ImageDataGenerator

from classifier.Classifier import Classifier
from commonsLib import loggerElk
from controller.ClassFile import ClassFile
from controller.Classify import Classify
from model.Configuration import Configuration
from model.Stats import Stats


class VisionClass:
    def __init__(self):
        self.conf = Configuration()
        self.logger = loggerElk(__name__, True)

    def _initialize(self, domain, model):
        self.logger.Information('GbcMlDocumentClassifierVisionClass::POST - loading configuration...')
        self.conf = Configuration()
        self.conf.working_path = os.path.join(self.conf.base_dir, domain, model)
        self.classify = Classify(self.conf)

    def _train_vision_data(self, data):
        # Loading dataset
        train_generator = ImageDataGenerator(rescale=1. / 255,
                                             shear_range=0.5,
                                             zoom_range=0.5,
                                             rotation_range=7)
        training_set = train_generator.flow_from_directory(
            os.path.join(self.conf.working_path, data, Classifier.TRAINING),
            target_size=(self.conf.nn_image_size, self.conf.nn_image_size),
            color_mode='grayscale',
            batch_size=self.conf.nn_batch_size,
            class_mode='categorical')

        validation_generator = ImageDataGenerator(rescale=1. / 255)
        validation_set = validation_generator.flow_from_directory(
            os.path.join(self.conf.working_path, data, Classifier.VALIDATION),
            color_mode='grayscale',
            target_size=(self.conf.nn_image_size, self.conf.nn_image_size),
            batch_size=self.conf.nn_batch_size,
            class_mode='categorical')

        return training_set, validation_set

    def train(self, domain, model, data):
        self._initialize(domain, model)

        stats = Stats()
        training_set, validation_set = self._train_vision_data(data)
        self.logger.Information(f'GbcMlDocumentClassifierVisionClass::POST - before training. model: {model}')

        if model == Classify.CNN_NETWORK:
            response = self.classify.launch_cnn_network(training_set, validation_set, None, True)
        else:
            response = []
            stats.info = 'Unsuitable training model. ' \
                         'Should be: CNN_NETWORK'
            stats.result = 'WRONG_MODEL'

        if len(response) == 3:
            stats.classifier = model
            stats.update_response(response)

        return stats

    def _predict_vision_data(self, data):
        test_generator = ImageDataGenerator(rescale=1. / 255)
        test_set = test_generator.flow_from_directory(self.conf.working_path,
                                                      target_size=(self.conf.nn_image_size, self.conf.nn_image_size),
                                                      color_mode='grayscale',
                                                      batch_size=self.conf.nn_batch_size,
                                                      class_mode=None,
                                                      shuffle=False)

        return test_set

    def predict(self, domain, model, data):
        self._initialize(domain, model)

        stats = Stats()
        prediction_set = self._predict_vision_data(data)
        self.logger.Information(f'GbcMlDocumentClassifierVisionClass::POST - before prediction. model: {model}')

        if model == Classify.CNN_NETWORK:
            response = self.classify.launch_cnn_network(None, None, prediction_set, False)
        else:
            response = []
            stats.info = 'Unsuitable prediction model. ' \
                         'Should be: CNN_NETWORK'
            stats.result = 'WRONG_MODEL'

        if len(response) == 3:
            stats.classifier = model
            stats.update_response(response)

        return stats

    @staticmethod
    def check(domain, model, data):
        conf = Configuration()
        if model == Classify.CNN_NETWORK:
            return ClassFile.has_media_file(os.path.join(conf.base_dir, domain, model, data))
        else:
            return False
