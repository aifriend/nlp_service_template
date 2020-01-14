import os
import shutil

import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.neural_network._multilayer_perceptron import BaseMultilayerPerceptron
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import check_X_y
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_is_fitted, column_or_1d
from tensorflow_core.python.keras import Sequential, callbacks
from tensorflow_core.python.keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow_core.python.keras.saving.save import save_model, load_model

from classifier.Classifier import Classifier
from commonsLib import loggerElk
from controller.ClassFile import ClassFile


class NNetwork(Classifier):
    class CNNClassifier(ClassifierMixin, BaseMultilayerPerceptron):

        def __init__(self, hidden_layer_sizes=(100,), activation="relu",
                     solver='adam', alpha=0.0001,
                     batch_size='auto', learning_rate="constant",
                     learning_rate_init=0.001, power_t=0.5, max_iter=200,
                     shuffle=True, random_state=None, tol=1e-4,
                     verbose=False, warm_start=False, momentum=0.9,
                     nesterovs_momentum=True, early_stopping=False,
                     validation_fraction=0.1, beta_1=0.9, beta_2=0.999,
                     epsilon=1e-8, n_iter_no_change=10, max_fun=15000, conf=None):
            super().__init__(
                hidden_layer_sizes=hidden_layer_sizes,
                activation=activation, solver=solver, alpha=alpha,
                batch_size=batch_size, learning_rate=learning_rate,
                learning_rate_init=learning_rate_init, power_t=power_t,
                max_iter=max_iter, loss='log_loss', shuffle=shuffle,
                random_state=random_state, tol=tol, verbose=verbose,
                warm_start=warm_start, momentum=momentum,
                nesterovs_momentum=nesterovs_momentum,
                early_stopping=early_stopping,
                validation_fraction=validation_fraction,
                beta_1=beta_1, beta_2=beta_2, epsilon=epsilon,
                n_iter_no_change=n_iter_no_change, max_fun=max_fun)
            # Load model
            self.conf = conf
            self.logger = loggerElk(__name__, True)

            # Building the model
            self.classifier = Sequential()

            # Creating the method for model
            # Step 1- Convolution
            self.classifier.add(Convolution2D(128, (5, 5),
                                              input_shape=(self.conf.nn_image_size, self.conf.nn_image_size, 1),
                                              activation='relu'))
            # adding another layer
            self.classifier.add(Convolution2D(64, (4, 4), activation='relu'))
            # Pooling it
            self.classifier.add(MaxPooling2D(pool_size=(2, 2)))
            # Adding another layer
            self.classifier.add(Convolution2D(32, (3, 3), activation='relu'))
            # Pooling
            self.classifier.add(MaxPooling2D(pool_size=(2, 2)))
            # Adding another layer
            self.classifier.add(Convolution2D(32, (3, 3), activation='relu'))
            # Pooling
            self.classifier.add(MaxPooling2D(pool_size=(2, 2)))
            # Step 2- Flattening
            self.classifier.add(Flatten())
            # Step 3- Full connection
            self.classifier.add(Dense(units=128, activation='relu'))
            # For the output step
            self.classifier.add(Dense(units=self.conf.nn_class_size, activation='softmax'))
            self.classifier.add(Dropout(0.02))
            # Add reularizers
            # classifier.add(Dense(128,
            #                input_dim = 128,
            #                kernel_regularizer = regularizers.l1(0.001),
            #                activity_regularizer = regularizers.l1(0.001),
            #                activation = 'relu'))

            self.classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

            # dropout = classifier.add(Dropout(0.2))

        def save_nn(self):
            try:
                dir_path = os.path.join(self.conf.working_path, self.conf.nn_model_name)
                if os.path.exists(dir_path):
                    shutil.rmtree(dir_path)
                os.makedirs(dir_path)
                save_model(self.classifier, filepath=dir_path, overwrite=True)
            except Exception as exc:
                self.logger.Error(exc)

        def load_nn(self):
            try:
                dir_path = os.path.join(self.conf.working_path, self.conf.nn_model_name)
                self.classifier = load_model(filepath=dir_path)
            except Exception as exc:
                self.logger.Error(exc)

        def fit(self, training_set, validation_set):
            """
            Fit the model to data matrix X and target(s) y.

            """
            check_pointer = callbacks.ModelCheckpoint(
                filepath=self.conf.working_path,
                monitor='val_acc',
                verbose=1,
                save_best_only=True,
                save_weights_only=False,
                mode='auto',
                period=1)
            history = self.classifier.fit_generator(training_set,
                                                    steps_per_epoch=(training_set.n / 32),
                                                    epochs=self.conf.nn_epochs,
                                                    validation_data=validation_set,
                                                    validation_steps=(validation_set.n / 32),
                                                    callbacks=[check_pointer])

        @property
        def partial_fit(self):
            """Update the model with a single iteration over the given data.

            classes : array, shape (n_classes), default None
                Classes across all calls to partial_fit.
                Can be obtained via `np.unique(y_all)`, where y_all is the
                target vector of the entire dataset.
                This argument is required for the first call to partial_fit
                and can be omitted in the subsequent calls.
                Note that y doesn't need to contain all labels in `classes`.

            Returns
            -------
            self : returns a trained MLP model.
            """
            # if self.solver not in _STOCHASTIC_SOLVERS:
            #     raise AttributeError("partial_fit is only available for stochastic"
            #                          " optimizer. %s is not stochastic"
            #                          % self.solver)
            # return self._partial_fit
            return

        def _partial_fit(self, X, y, classes=None):
            # if _check_partial_fit_first_call(self, classes):
            #     self._label_binarizer = LabelBinarizer()
            #     if type_of_target(y).startswith('multilabel'):
            #         self._label_binarizer.fit(y)
            #     else:
            #         self._label_binarizer.fit(classes)
            #
            # super()._partial_fit(X, y)
            #
            # return self
            pass

        def _validate_input(self, X, y, incremental):
            X, y = check_X_y(X, y, accept_sparse=['csr', 'csc', 'coo'],
                             multi_output=True)
            if y.ndim == 2 and y.shape[1] == 1:
                y = column_or_1d(y, warn=True)

            if not incremental:
                self._label_binarizer = LabelBinarizer()
                self._label_binarizer.fit(y)
                self.classes_ = self._label_binarizer.classes_
            elif self.warm_start:
                classes = unique_labels(y)
                if set(classes) != set(self.classes_):
                    raise ValueError("warm_start can only be used where `y` has "
                                     "the same classes as in the previous "
                                     "call to fit. Previously got %s, `y` has %s" %
                                     (self.classes_, classes))
            else:
                classes = unique_labels(y)
                if len(np.setdiff1d(classes, self.classes_, assume_unique=True)):
                    raise ValueError("`y` has classes not in `self.classes_`."
                                     " `self.classes_` has %s. 'y' has %s." %
                                     (self.classes_, classes))

            y = self._label_binarizer.transform(y)
            return X, y

        def predict(self, X):
            """Predict using the multi-layer perceptron classifier

            Parameters
            ----------
            X : {array-like, sparse matrix}, shape (n_samples, n_features)
                The input data.

            Returns
            -------
            y : array-like, shape (n_samples,) or (n_samples, n_classes)
                The predicted classes.
            """
            # check_is_fitted(self)
            # y_pred = self._predict(X)
            #
            # if self.n_outputs_ == 1:
            #     y_pred = y_pred.ravel()
            #
            # return self._label_binarizer.inverse_transform(y_pred)

            y_pred = self.classifier.predict(X)
            return y_pred

        def predict_log_proba(self, X):
            """Return the log of probability estimates.

            Parameters
            ----------
            X : array-like, shape (n_samples, n_features)
                The input data.

            Returns
            -------
            log_y_prob : array-like, shape (n_samples, n_classes)
                The predicted log-probability of the sample for each class
                in the model, where classes are ordered as they are in
                `self.classes_`. Equivalent to log(predict_proba(X))
            """
            # y_prob = self.predict_proba(X)
            # return np.log(y_prob, out=y_prob)
            pass

        def predict_proba(self, X):
            """Probability estimates.

            Parameters
            ----------
            X : {array-like, sparse matrix}, shape (n_samples, n_features)
                The input data.

            Returns
            -------
            y_prob : array-like, shape (n_samples, n_classes)
                The predicted probability of the sample for each class in the
                model, where classes are ordered as they are in `self.classes_`.
            """
            check_is_fitted(self)
            y_pred = self.classifier.predict_proba(X)

            if self.n_outputs_ == 1:
                y_pred = y_pred.ravel()

            if y_pred.ndim == 1:
                return np.vstack([1 - y_pred, y_pred]).T
            else:
                return y_pred

    def __init__(self, conf):
        self.conf = conf
        super().__init__(self.conf)

    def initialize(self):
        self.clf = self.CNNClassifier(solver=self.conf.nn_solver,
                                      alpha=self.conf.nn_alpha,
                                      hidden_layer_sizes=self.conf.nn_hidden_layer_sizes,
                                      random_state=self.conf.nn_random_state,
                                      conf=self.conf)

    def train(self, x, y):
        self.initialize()
        self.do_train(x, y)
        self.save_model()

    def predict(self, x):
        self.initialize()
        self.load_model()

        y = None
        response = list()
        try:
            class_path = os.path.join(self.conf.base_dir, self.conf.classes)
            class_list = ClassFile.file_to_list(class_path, binary=False)
            predicted = self.clf.classifier.predict_generator(x, verbose=1, steps=(x.n / 32))
            predicted_class_indices = np.argmax(predicted, axis=1)
            predictions = [class_list[k] for k in predicted_class_indices]

            response.append(predicted)
            response.append(class_list)
            response.append(predictions)
        except Exception as exc:
            pass

        return response

    def predict_probs(self, x, model=None):
        self.initialize()
        self.load_model()
        classes, probs = self.do_predict_probs(x, model)
        return classes, probs

    def load_model(self, **kwargs):
        self.clf.load_nn()

    def save_model(self, **kwargs):
        self.clf.save_nn()
