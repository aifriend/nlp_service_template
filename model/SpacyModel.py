import en_core_web_md
import es_core_news_md

from commonsLib import loggerElk

logger = loggerElk(__name__, True)


class SpacyModel:
    # Here will be the instance stored.
    __instance = None
    model = dict()

    @staticmethod
    def getInstance():
        """ Static access method. """
        if SpacyModel.__instance is None:
            SpacyModel()
        return SpacyModel.__instance

    def getModel(self, lang):
        return self.model[lang]

    def __init__(self):
        """ Virtually private constructor. """
        if SpacyModel.__instance is not None:
            raise Exception("This class is a singleton!")
        else:
            logger.Information('GbcMlDocumentClassifierPrediction::POST.SpacyModel - loading spacy\'s model...')
            self.model['es'] = es_core_news_md.load()
            self.model['en'] = en_core_web_md.load()
            logger.Information('GbcMlDocumentClassifierPrediction::POST.SpacyModel - ...loaded')

            SpacyModel.__instance = self


def main():
    '''
    s = Singleton.getInstance()
    o = Singleton.getInstance()
    print(len(s.vectorizers))
    print(len(o.vectorizers))
    print(o.vectorizers)
    '''


if __name__ == '__main__':
    main()
