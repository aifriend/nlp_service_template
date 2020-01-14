from sklearn.ensemble import VotingClassifier

from classifier.Classifier import Classifier


class Voting(Classifier):

    def __init__(self, conf):
        self.conf = conf
        super().__init__(self.conf)

    def initialize(self, clf_list):
        # ('rf', clf_rf), ('nb', clf_nb), ('dt', clf_dt)
        self.clf = VotingClassifier(
            estimators=clf_list,
            voting=self.conf.voting,
            n_jobs=self.conf.voting_n_jobs)

    def train(self, feature, label, class_list):
        self.initialize(class_list)
        self.do_train(feature, label)
        self.save_model(self.conf.voting_model)

    def predict(self, x, class_list):
        self.initialize(class_list)
        self.load_model()

        response = list()
        predicted = list()
        class_list = dict()
        probability_list = self.do_predict(x).tolist()
        thresh = 0.5
        for i, prob in enumerate(probability_list):
            for ci_k in class_list.keys():
                if prob[ci_k] > thresh:
                    predicted.append(class_list[ci_k])

        response.append(predicted)
        response.append(class_list)
        response.append(probability_list)

        return response

    def predict_probs(self, x, class_list, model=None):
        self.initialize(class_list)
        self.load_model()
        classes, probs = self.do_predict_probs(x, model)
        return classes, probs

    def load_model(self, path=''):
        super().load_model(self.conf.voting_model)
