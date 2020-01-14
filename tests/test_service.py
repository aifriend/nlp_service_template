import logging
import os
import unittest
from contextlib import contextmanager
from tempfile import mkstemp

from binaryornot.check import is_binary
from hypothesis import given
from hypothesis.strategies import binary

from controller.Classify import Classify
from tests.ITest import ITest

logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

SERVER_URL = "http://{0}:{1}/".format('127.0.0.1', 7116)
DATA_URL = "d:/work/train/svh_mini/"
TRAINING_URL = "api/gbc/ml/document/classifier/training"
PREDICTION_URL = "api/gbc/ml/document/classifier/predict"


@contextmanager
def bytes_in_file(data):
    o, f = mkstemp()
    try:
        os.write(o, data)
        os.close(o)
        yield f
    finally:
        os.unlink(f)


class TestDetectionProperties(unittest.TestCase):
    @given(binary(max_size=512))
    def test_never_crashes(self, data):
        with bytes_in_file(data) as f:
            is_binary(f)


class TestServer(unittest.TestCase):
    def test_up(self):
        self.assertTrue(ITest.local_server_up(SERVER_URL))


class TestDoTraining(unittest.TestCase):
    def test_image_training(self):
        self.assertTrue(
            ITest.do_request(
                server_url=SERVER_URL + TRAINING_URL,
                source="IMAGE",
                data_path=DATA_URL,
                model=Classify.CNN_NETWORK) is not None
        )

    def test_text_training_plain(self):
        self.assertTrue(
            ITest.do_request(
                server_url=SERVER_URL + TRAINING_URL,
                source="PLAINTEXT",
                data_path=DATA_URL,
                model=Classify.NAIVE_BAYES_MULTI) is not None
        )

    def test_text_training_file_bagging(self):
        self.assertTrue(
            ITest.do_request(
                server_url=SERVER_URL + TRAINING_URL,
                source="FILE",
                data_path=DATA_URL,
                model=Classify.BAGGING) is not None
        )

    def test_text_training_file_boosting_ada(self):
        self.assertTrue(
            ITest.do_request(
                server_url=SERVER_URL + TRAINING_URL,
                source="FILE",
                data_path=DATA_URL,
                model=Classify.BOOSTING_ADA) is not None
        )

    def test_text_training_file_boosting_sgd(self):
        self.assertTrue(
            ITest.do_request(
                server_url=SERVER_URL + TRAINING_URL,
                source="FILE",
                data_path=DATA_URL,
                model=Classify.BOOSTING_SGD) is not None
        )

    def test_text_training_file_decision_tree(self):
        self.assertTrue(
            ITest.do_request(
                server_url=SERVER_URL + TRAINING_URL,
                source="FILE",
                data_path=DATA_URL,
                model=Classify.DECISION_TREE) is not None
        )

    def test_text_training_file_extra_tree(self):
        self.assertTrue(
            ITest.do_request(
                server_url=SERVER_URL + TRAINING_URL,
                source="FILE",
                data_path=DATA_URL,
                model=Classify.EXTRA_TREES) is not None
        )

    def test_text_training_file_bayes_multi(self):
        self.assertTrue(
            ITest.do_request(
                server_url=SERVER_URL + TRAINING_URL,
                source="FILE",
                data_path=DATA_URL,
                model=Classify.NAIVE_BAYES_MULTI) is not None
        )

    def test_text_training_file_bayes_complement(self):
        self.assertTrue(
            ITest.do_request(
                server_url=SERVER_URL + TRAINING_URL,
                source="FILE",
                data_path=DATA_URL,
                model=Classify.NAIVE_BAYES_COMPLEMENT) is not None
        )

    def test_text_training_file_random_forest(self):
        self.assertTrue(
            ITest.do_request(
                server_url=SERVER_URL + TRAINING_URL,
                source="FILE",
                data_path=DATA_URL,
                model=Classify.RANDOM_FOREST) is not None
        )

    def test_text_training_file_voting(self):
        self.assertTrue(
            ITest.do_request(
                server_url=SERVER_URL + TRAINING_URL,
                source="FILE",
                data_path=DATA_URL,
                model="VOTING") is not None
        )


class TestDoPrediction(unittest.TestCase):
    def test_image_prediction(self):
        self.assertTrue(
            ITest.do_request(
                server_url=SERVER_URL + PREDICTION_URL,
                source="IMAGE",
                data_path=DATA_URL,
                model=Classify.CNN_NETWORK) is not None
        )

    def test_text_prediction_plain(self):
        self.assertTrue(
            ITest.do_request(
                server_url=SERVER_URL + PREDICTION_URL,
                source="PLAINTEXT",
                data_path=DATA_URL,
                model=Classify.NAIVE_BAYES_MULTI) is not None
        )

    def test_text_prediction_file_bagging(self):
        self.assertTrue(
            ITest.do_request(
                server_url=SERVER_URL + PREDICTION_URL,
                source="FILE",
                data_path=DATA_URL,
                model=Classify.BAGGING) is not None
        )

    def test_text_prediction_file_boosting_ada(self):
        self.assertTrue(
            ITest.do_request(
                server_url=SERVER_URL + PREDICTION_URL,
                source="FILE",
                data_path=DATA_URL,
                model=Classify.BOOSTING_ADA) is not None
        )

    def test_text_prediction_file_boosting_sgd(self):
        self.assertTrue(
            ITest.do_request(
                server_url=SERVER_URL + PREDICTION_URL,
                source="FILE",
                data_path=DATA_URL,
                model=Classify.BOOSTING_SGD) is not None
        )

    def test_text_prediction_file_decision_tree(self):
        self.assertTrue(
            ITest.do_request(
                server_url=SERVER_URL + PREDICTION_URL,
                source="FILE",
                data_path=DATA_URL,
                model=Classify.DECISION_TREE) is not None
        )

    def test_text_prediction_file_extra_trees(self):
        self.assertTrue(
            ITest.do_request(
                server_url=SERVER_URL + PREDICTION_URL,
                source="FILE",
                data_path=DATA_URL,
                model=Classify.EXTRA_TREES) is not None
        )

    def test_text_prediction_file_bayes_multi(self):
        self.assertTrue(
            ITest.do_request(
                server_url=SERVER_URL + PREDICTION_URL,
                source="FILE",
                data_path=DATA_URL,
                model=Classify.NAIVE_BAYES_MULTI) is not None
        )

    def test_text_prediction_file_bayes_complement(self):
        self.assertTrue(
            ITest.do_request(
                server_url=SERVER_URL + PREDICTION_URL,
                source="FILE",
                data_path=DATA_URL,
                model=Classify.NAIVE_BAYES_COMPLEMENT) is not None
        )

    def test_text_prediction_file_random_forest(self):
        self.assertTrue(
            ITest.do_request(
                server_url=SERVER_URL + PREDICTION_URL,
                source="FILE",
                data_path=DATA_URL,
                model=Classify.RANDOM_FOREST) is not None
        )

    def test_text_prediction_file_voting(self):
        self.assertTrue(
            ITest.do_request(
                server_url=SERVER_URL + PREDICTION_URL,
                source="FILE",
                data_path=DATA_URL,
                model="VOTING") is not None
        )


if __name__ == '__main__':
    unittest.main()
