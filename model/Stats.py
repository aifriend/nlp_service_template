import json


class Stats:

    def __init__(self,
                 classes: list = None, classifier: str = '',
                 info: str = '', predicted: list = None,
                 probabilities: list = None, result: str = 'OK'):
        self.classifier = classifier
        self.result = result
        self.info = info
        self.predicted = None
        self.classes = None
        self.probabilities = None

    def update_response(self, response):
        if response is not None and len(response) == 3:
            self.predicted = response[0]
            self.classes = response[1]
            self.probabilities = response[2]

    def to_json(self):
        return json.loads(json.dumps(self.__dict__))
