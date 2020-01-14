import os

import numpy as np
from service import Configuration
from service import Stats
from sklearn.model_selection import train_test_split

from commonsLib import loggerElk
from controller.ClassFile import ClassFile
from controller.Classify import Classify
from controller.PreProcess import PreProcess


class TextClass:

    def __init__(self):
        self.conf = Configuration()
        self.logger = loggerElk(__name__, True)

    def _initialize(self, domain, model, nlp=None, dictionary=None):
        self.logger.Information('GbcMlDocumentClassifierTextClass::POST - loading configuration...')
        self.conf.working_path = os.path.join(self.conf.base_dir, domain, model)
        if dictionary is not None:
            self.conf.dictionary = dictionary
        if nlp is not None:
            self.logger.Information('GbcMlDocumentClassifierTextClass::POST - pre-processing...')
            self.pre_process = PreProcess(self.conf, nlp)
        self.classify = Classify(self.conf)

    def train(self, domain, model, data, nlp, dictionary):
        self._initialize(domain, model, nlp, dictionary)

        stats = Stats()
        training_set = os.path.join(self.conf.working_path, data)
        v_list = self.classify.from_file.list_files_ext(training_set, ".gram")
        X = []
        y = []
        corpus_size = len(v_list)
        vector_size = 0
        for f in v_list:
            vector = self.pre_process.get_tfidf_from_vectorizer(self.classify.from_file.file_to_list(f))
            category = self.classify.get_category(f)
            n_vector = vector.toarray()[0]
            vector_size = len(n_vector)
            X.append(n_vector)
            y.append(category)
        X = np.array(X).reshape((corpus_size, vector_size))
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=11)
        self.logger.Information(f'GbcMlDocumentClassifierTextClass::POST - before training. model: {model}')

        response = []
        if nlp:
            if model == Classify.BAGGING:
                response = self.classify.launch_bagging_classifier(X_train, y_train, None, True)
                self.logger.Information(self.classify.show_metrics(y_test, response[0], stats=stats))
            elif model == Classify.BOOSTING_ADA:
                response = self.classify.launch_boosting_ada(X_train, y_train, None, True)
                self.logger.Information(self.classify.show_metrics(y_test, response[0], stats=stats))
            elif model == Classify.BOOSTING_SGD:
                response = self.classify.launch_boosting_sgd(X_train, y_train, None, True)
                self.logger.Information(self.classify.show_metrics(y_test, response[0], stats=stats))
            elif model == Classify.DECISION_TREE:
                response = self.classify.launch_decision_tree(X_train, y_train, None, True)
                self.logger.Information(self.classify.show_metrics(y_test, response[0], stats=stats))
            elif model == Classify.EXTRA_TREES:
                response = self.classify.launch_extra_trees(X_train, y_train, None, True)
                self.logger.Information(self.classify.show_metrics(y_test, response[0], stats=stats))
            elif model == Classify.NAIVE_BAYES_MULTI:
                response = self.classify.launch_naive_bayes_multinomial(X_train, y_train, None, True)
                self.logger.Information(self.classify.show_metrics(y_test, response[0], stats=stats))
            elif model == Classify.NAIVE_BAYES_COMPLEMENT:
                response = self.classify.launch_naive_bayes_complement(X_train, y_train, None, True)
                self.logger.Information(self.classify.show_metrics(y_test, response[0], stats=stats))
            elif model == Classify.RANDOM_FOREST:
                response = self.classify.launch_random_forest(X_train, y_train, None, True)
                self.logger.Information(self.classify.show_metrics(y_test, response[0], stats=stats))
            elif model == Classify.VOTING:
                response = self.classify.launch_voting_classifier(X_train, y_train, None, True)
                self.logger.Information(self.classify.show_metrics(y_test, response[0], stats=stats))
            else:
                response = []
                stats.info = 'Unsuitable training model. ' \
                             'Should be one of: BAGGING | BOOSTING_ADA | BOOSTING_SGD ' \
                             '| DECISION_TREE | EXTRA_TREES | NAIVE_BAYES_MULTI ' \
                             '| NAIVE_BAYES_COMPLEMENT | RANDOM_FOREST | VOTING'
                stats.result = 'WRONG_MODEL'

        if len(response) == 3:
            stats.classifier = model
            stats.update_response(response)

        return stats

    def predict(self, domain, model, nlp, data=None, file=None):
        self._initialize(domain, model, nlp)

        stats = Stats()
        response = []
        if nlp:
            if data is None and file:
                vector = self.pre_process.transform(domain, file)
            else:
                vector = self.pre_process.transform_text(data)
            # sparse_vector = SparseVector()
            # sparse_vector.fromList([x for x in vector.toarray()[0]])
            # self.logger.Information(sparse_vector)

            X = [vector.toarray()[0]]
            X = np.array(X).reshape((1, len(vector.toarray()[0])))
            self.logger.Information(f'GbcMlDocumentClassifierTextClass::POST - before prediction. model: {model}')

            if model == Classify.NAIVE_BAYES_MULTI:
                response = self.classify.launch_naive_bayes_multinomial(None, None, X, False)
                self.logger.Information(self.classify.show_metrics(X, response[0], stats=stats))
            elif model == Classify.VOTING:
                response = self.classify.launch_voting_classifier(None, None, X, False)
                self.logger.Information(self.classify.show_metrics(X, response[0], stats=stats))
            else:
                response = []
                stats.info = 'Unsuitable prediction model. ' \
                             'Should be one of: BAGGING | BOOSTING_ADA | BOOSTING_SGD ' \
                             '| DECISION_TREE | EXTRA_TREES | NAIVE_BAYES_MULTI ' \
                             '| NAIVE_BAYES_COMPLEMENT | RANDOM_FOREST | VOTING )'
                stats.result = 'WRONG_MODEL'

        if len(response) == 3:
            stats.classifier = model
            stats.update_response(response)

        return stats

    @staticmethod
    def check(domain, model, data):
        conf = Configuration()
        if model == Classify.NAIVE_BAYES_MULTI or model == Classify.VOTING:
            return ClassFile.has_text_file(os.path.join(conf.base_dir, domain, model, data))
        else:
            return False
