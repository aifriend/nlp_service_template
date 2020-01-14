import os

import yaml

from commonsLib import loggerElk
from controller.ClassFile import ClassFile


class Configuration:

    def __init__(self, conf_route="model/config.yml"):
        self.logger = loggerElk(__name__, True)

        # load config file
        try:
            with open(conf_route, 'r') as yml_file:
                cfg = yaml.load(yml_file)
        except Exception as exc:
            self.logger.Error("Error loading model/config.yml...")

        development = cfg['development']

        # data augmentation
        self.examples_per_case = development['augmentation']

        # server
        server = development['server']
        self.server_host = server['host']
        self.server_port = server['port']

        # image processing
        image_processing = development['image_processing']
        self.resize_width = image_processing['resize_width']
        self.resize_height = image_processing['resize_height']
        self.crop_width = image_processing['crop_width']
        self.crop_height = image_processing['crop_height']

        # working directories
        directories = development['directories']
        self.base_dir = directories['base_dir']
        self.sample = directories['sample']
        self.cat_file = directories['cat_file']
        self.dictionary = directories['dictionary']
        self.classes = directories['classes']
        self.working_path = ''

        # pre process
        pre_process = development['pre_process']
        self.pre_process_batch_size = pre_process['pre_process_batch_size']
        self.max_string_size = pre_process['max_string_size']
        self.spa_dict = []

        # files
        files = development['files']
        self.tf = files['tf']
        self.tfidf = files['tfidf']
        self.vectorizer = files['vectorizer']

        # system separator
        self.sep = os.path.sep

        # cnn layers sizes
        nn_layer = development['nn_layer']
        self.filters = nn_layer['filters']
        self.kernels = nn_layer['kernels']
        self.units = nn_layer['units']
        self.epochs = nn_layer['epochs']

        # bagging
        bagging = development['bagging']
        self.bagging_model = bagging['bagging_model']
        self.bagging_n_estimators = bagging['bagging_n_estimators']
        self.bagging_max_samples = bagging['bagging_max_samples']
        self.bagging_bootstrap = bagging['bagging_bootstrap']
        self.bagging_n_jobs = bagging['bagging_n_jobs']

        # boosting
        boosting = development['boosting']
        self.boosting_model = boosting['boosting_model']
        self.boosting_n_estimators = boosting['boosting_n_estimators']

        # decision tree
        decision_tree = development['decision tree']
        self.dt_model = decision_tree['dt_model']
        self.dt_max_depth = decision_tree['dt_max_depth']

        # extra trees
        extra_trees = development['extra trees']
        self.et_model = extra_trees['et_model']
        self.et_n_estimators = extra_trees['et_n_estimators']
        self.et_max_features = extra_trees['et_max_features']
        self.et_bootstrap = extra_trees['et_bootstrap']
        self.et_n_jobs = extra_trees['et_n_jobs']

        # naive bayes
        naive_bayes = development['naive bayes']
        self.nb_model = naive_bayes['nb_model']

        # random forest
        random_forest = development['random forest']
        self.rf_model = random_forest['rf_model']
        self.rf_n_estimators = random_forest['rf_n_estimators']
        self.rf_max_leaf_nodes = random_forest['rf_max_leaf_nodes']
        self.rf_n_jobs = random_forest['rf_n_jobs']

        # n_network
        n_network = development['nnetwork']
        self.nn_model_name = n_network['nn_model_name']
        self.nn_solver = n_network['nn_solver']
        self.nn_alpha = n_network['nn_alpha']
        self.nn_hidden_layer_sizes = n_network['nn_hidden_layer_sizes']
        self.nn_random_state = n_network['nn_random_state']
        self.nn_image_size = n_network['nn_image_size']
        self.nn_class_size = n_network['nn_class_size']
        self.nn_batch_size = n_network['nn_batch_size']
        self.nn_epochs = n_network['nn_epochs']

        # voting
        voting = development['voting']
        self.voting_model = voting['voting_model']
        self.voting = voting['voting']
        self.voting_n_jobs = voting['voting_n_jobs']

    def load_dict(self):
        print(os.getcwd())
        dict_file = ClassFile.get_text(self.dictionary).split('\n')
        dict_file = filter(lambda x: (x is not ''), dict_file)
        spa_dict = list(map(lambda x: x.lower(), dict_file))
        self.spa_dict = spa_dict

        return spa_dict
