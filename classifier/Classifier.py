import os

from commonsLib import loggerElk
from controller.ClassFile import ClassFile
from model.Configuration import Configuration


class Classifier(object):
    TRAINING = "training"
    VALIDATION = "validation"

    def __init__(self, conf: Configuration):
        self.conf = conf
        self.clf = None
        self.logger = loggerElk(__name__, True)

    def initialize(self, *args):
        pass

    def do_train(self, x, y):
        self.clf.fit(x, y)

    def do_predict(self, x):
        if self.clf is None:
            print("Please, load model")
            return None

        y = self.clf.predict(x)
        return y

    def do_predict_probs(self, x, model=None):
        if model is not None:
            self.clf = model

        if self.clf is None:
            print("Please, load model")
            return None

        y = self.clf.predict_proba(x)
        return self.clf.classes_, y

    def load_model(self, model_name):
        try:
            path = os.path.join(self.conf.working_path, model_name)
            self.clf = ClassFile.load_model(path)
        except Exception as exc:
            self.logger.Error("Model file not found")

    def save_model(self, model_name):
        try:
            path = os.path.join(self.conf.working_path, model_name)
            ClassFile.save_model(path, self.clf)
        except Exception as exc:
            self.logger.Error("Model file not saved")
