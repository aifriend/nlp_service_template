from commonsLib import loggerElk
from controller.ClassFile import ClassFile

logger = loggerElk(__name__, True)


class Singleton:
    # Here will be the instance stored.
    __instance = None
    vectorizers = None

    @staticmethod
    def getInstance(conf):
        """ Static access method. """
        if Singleton.__instance is None:
            Singleton(conf)
        return Singleton.__instance

    def __init__(self, conf):
        """ Virtually private constructor. """
        if Singleton.__instance is not None:
            raise Exception("This class is a singleton!")
        else:
            self.vectorizers = {}

            files = ClassFile.list_files_ext(conf.base_dir, 'vectorizer.tfidf')
            logger.Information('GbcMlDocumentClassifierPrediction::POST - loading vectorizers...')

            for f in files:
                key = ClassFile.get_containing_dir_name(f)
                logger.Information(f'GbcMlDocumentClassifierPrediction::POST - loading model: {key}...')
                self.vectorizers[key] = ClassFile.load_model(f)
                logger.Information(f'GbcMlDocumentClassifierPrediction::POST - loaded model: ...{key}')

            Singleton.__instance = self


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
