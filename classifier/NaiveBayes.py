from sklearn.naive_bayes import MultinomialNB, ComplementNB

from classifier.Classifier import Classifier


class NaiveBayes(Classifier):

    def __init__(self, conf):
        self.conf = conf
        super().__init__(self.conf)

    def initialize(self, subtype='multinomial'):
        if subtype is 'complement':
            self.clf = ComplementNB()
        elif subtype is 'multinomial':
            self.clf = MultinomialNB()
        else:
            print("Unknown Naive-Bayes type")

    def train(self, x, y, subtype='multinomial'):
        self.initialize(subtype=subtype)
        self.do_train(x, y)
        self.save_model(self.conf.nb_model.replace('(subtype)', subtype))

    def predict(self, x, subtype='multinomial'):
        self.initialize(subtype=subtype)
        self.load_model(subtype)

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

    def predict_probs(self, x, subtype='multinomial', model=None):
        self.initialize(subtype=subtype)
        self.load_model(subtype)
        classes, probs = self.do_predict_probs(x, model)
        return classes, probs

    def load_model(self, subtype='multinomial'):
        super().load_model(self.conf.nb_model.replace('(subtype)', subtype))
