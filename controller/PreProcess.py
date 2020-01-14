import os
from queue import Queue
from threading import Thread

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer

from commonsLib import loggerElk
from controller.ClassFile import ClassFile
from model.Configuration import Configuration
from model.Document import Document
from model.Sentence import Sentence
from model.Singleton import Singleton
from model.SpacyModel import SpacyModel
from nlp import utils, stopwords


class PreProcess:

    def __init__(self, conf, nlp):
        self.conf = conf
        self.nlp = nlp
        self.logger.Information('GbcMlDocumentClassifierPrediction::POST - loading dictionary...')
        self.conf.load_dict()
        self.tf = None
        self.tf_idf = None
        self.vectorizer = None
        self.from_file = ClassFile()

        self.logger = loggerElk(__name__, True)

    def process(self, text, kind='none', path=''):
        """
        process svh texts

        """
        xdoc = Document()
        xdoc.kind = kind
        xdoc.path = path

        doc = None
        sentences = []
        if len(text) > self.conf.max_string_size:
            print(len(text))
            split = utils.split_by_size(text, self.conf.max_string_size)
            for t in split:
                doc = self.nlp(t)
                for s in doc.sents:
                    sentences.append(s)
        else:
            doc = self.nlp(text)
            sentences = doc.sents

        # mark stopwords
        for sentence in doc.sents:
            # print(sentence)
            s = Sentence()
            for token in sentence:
                t = stopwords.clean_token(self.conf, token)
                s.add_token(t)

            xdoc.add_sentence(s)

        # build 1-grams (just lemmas)
        for s in xdoc.sentences:
            for t in s.tokens:
                if not t.stop:
                    xdoc.add_gram(t.lemma.lower())

        # print(xdoc.path, ':\n', ' '.join(sorted(list(xdoc.grams))))
        self.from_file.list_to_file(list(xdoc.grams), self.from_file.file_base_name(xdoc.path) + '.gram')
        # print('', ' '.join(files.file_to_list(files.file_base_name(xdoc.path) + '.gram')))
        return xdoc

    def load_vector_models(self):
        self.tf = self.from_file.load_model(os.path.join(self.conf.working_path, self.conf.tf))
        self.tf_idf = self.from_file.load_model(os.path.join(self.conf.working_path, self.conf.tfidf))
        # print(self.tf.vocabulary_)

    def load_vectorizer_model(self, domain):
        self.logger.Information('GbcMlDocumentClassifierPrediction::POST - transform...')
        self.vectorizer = Singleton.getInstance(self.conf).vectorizers[domain]

    def get_tfidf(self, gram):
        count = self.tf.transform([' '.join(gram)])
        vector = self.tf_idf.transform(count)
        # print(vector.toarray()[0].tolist())
        return vector

    def get_tfidf_from_vectorizer(self, gram):
        vector = self.vectorizer.transform([' '.join(gram)])
        # print(vector.toarray()[0].tolist())
        return vector

    def get_count(self, gram):
        count = self.tf.transform([' '.join(gram)])
        # print(count.shape)
        return count

    def transform(self, domain, file):
        if self.vectorizer is None:
            self.load_vectorizer_model(domain)
        text = self.from_file.get_text(file)
        return self.transform_text(text)

    def transform_text(self, text):
        doc = self.process(utils.clean_text(text), 'none')
        # print(doc.grams)
        vector = self.get_tfidf_from_vectorizer(doc.grams)
        # X = [vector.toarray()[0]]
        # X = np.array(X).reshape((1, len(vector.toarray()[0])))
        return vector

    def _do_pre_process(self, q, result):  # q:[[index, text, kind, path], ...]
        """
        launch svh text processing in threads

        """
        while not q.empty():
            work = q.get()  # fetch new work from the Queue
            try:
                print("Requested..." + str(work[0]))
                data = self.process(work[1], work[2], work[3])
                result[work[0]] = data  # Store data back at correct index
                print(".............................. Done " + str(work[0]))
            except Exception as exc:
                result[work[0]] = Document()
                self.logger.Error(exc)

            # signal to the queue that task has been processed
            q.task_done()
        return True

    def _create_dataset(self, docs):
        """
        build the tf and tfidf matrixes for the whole svh text

        """
        text = []
        for doc in docs:
            text.append(doc.get_grams_as_text())

        # create the transform
        # tokenize and build vocab
        count_vectorizer = CountVectorizer()
        x_tf = count_vectorizer.fit_transform(text)
        print(x_tf.shape)

        # idf
        tfidf_transformer = TfidfTransformer()
        x_tfidf = tfidf_transformer.fit_transform(x_tf)
        print(x_tfidf.shape)

        # encode documents
        for doc in docs:
            vector_tf = count_vectorizer.transform([doc.get_grams_as_text()])
            print(vector_tf.shape)
            print(type(vector_tf))
            print(vector_tf.toarray())
            self.from_file.save_sparse_csr(self.from_file.file_base_name(doc.path) + '.tf', vector_tf)

            vector_tfidf = tfidf_transformer.transform(vector_tf)
            print(vector_tfidf.shape)
            print(type(vector_tfidf))
            print(vector_tfidf.toarray())
            self.from_file.save_sparse_csr(self.from_file.file_base_name(doc.path) + '.tfidf', vector_tfidf)

        return x_tf, x_tfidf

    def create_dataset_from_unigrams_direct(self, uni_grams):
        text = []
        for doc_grams in uni_grams:
            if len(doc_grams) == 0:
                print('.< size 0 vector >.')
            else:
                text.append(' '.join(list(doc_grams)))

        # create the transform
        vectorizer = TfidfVectorizer(min_df=1, max_df=0.99)
        x_tfidf = vectorizer.fit_transform(text)
        print('tfidf shape:', x_tfidf.shape)

        self.from_file.save_model(os.path.join(self.conf.working_path, 'vectorizer.tfidf'), vectorizer)

        return vectorizer

    def _create_dataset_from_uni_grams(self, uni_grams):
        """
        Build the tf and tfidf matrixes for the whole svh text loading all .gram files

        """
        text = []
        for doc_grams in uni_grams:
            if len(doc_grams) == 0:
                print('.< size 0 vector >.')
            else:
                text.append(' '.join(list(doc_grams)))

        # create the transform
        # tokenize and build vocab
        count_vectorizer = CountVectorizer()
        x_tf = count_vectorizer.fit_transform(text)
        print(x_tf.shape)

        # idf
        tfidf_transformer = TfidfTransformer()
        x_tfidf = tfidf_transformer.fit_transform(x_tf)
        print(x_tfidf.shape)

        self.from_file.save_model(os.path.join(self.conf.working_path, self.conf.tf), count_vectorizer)
        self.from_file.save_model(os.path.join(self.conf.working_path, self.conf.tfidf), tfidf_transformer)

        return x_tf, x_tfidf

    def pre_process_batches(self):
        """
        Process all svh txt files in batches to get the .grams

        """
        categories = set()
        all_categories = []

        d_list = self.from_file.list_files_ext(self.conf.working_path, ".txt")
        all_docs = [None for d in d_list]
        q = Queue(maxsize=0)

        counter = 0
        total = len(d_list)
        i = 0

        while i < total:
            h = i
            for j in range(self.conf.pre_process_batch_size):
                if h < total:
                    f = d_list[h]
                    category = self.from_file.get_containing_dir_name(f)
                    categories.add(category)

                    all_categories.append(category)
                    text = utils.clean_text(self.from_file.get_text(f))

                    print('doc %s to q' % (counter + 1))
                    q.put((counter, text, category, f))

                    counter += 1
                h += 1

            for j in range(q.qsize()):
                worker = Thread(target=self._do_pre_process, args=(q, all_docs))
                worker.setDaemon(True)  # setting threads as "daemon" allows main program to
                # exit eventually even if these dont finish
                # correctly.
                worker.start()

            # now we wait until the queue has been processed
            q.join()

            q.empty()
            i = h

        print(len(categories), categories)
        # create_dataset(conf, all_docs)

    def create_full_dataset_vectorizer(self):
        """
        Load all .gram files and call create_dataset_from_unigrams

        """
        v_list = self.from_file.list_files_ext(self.conf.working_path, ".gram")
        unigrams = []
        print(v_list)

        for f in v_list:
            unigrams.append(self.from_file.file_to_list(f))

        self.create_dataset_from_unigrams_direct(unigrams)

    def _pre_process(self):
        """
        Process all svh txt files to get the .grams

        """
        categories = set()
        all_categories = []

        d_list = self.from_file.list_files_ext(self.conf.working_path, "txt")
        all_docs = [None for d in d_list]
        q = Queue(maxsize=0)

        counter = 0
        total = len(d_list)
        cumul = 0
        for f in d_list:
            category = self.from_file.get_containing_dir_name(f)
            categories.add(category)

            all_categories.append(category)
            text = utils.clean_text(self.from_file.get_text(f))

            print('doc %s to q' % (counter + 1))
            q.put((counter, text, category, f))

            counter += 1

        for i in range(total):
            worker = Thread(target=self._do_pre_process, args=(q, all_docs))
            worker.setDaemon(True)  # setting threads as "daemon" allows main program to
            # exit eventually even if these dont finish
            # correctly.
            worker.start()
        # now we wait until the queue has been processed
        q.join()

        print(len(categories), categories)
        # create_dataset(conf, all_docs)

    @staticmethod
    def test_pre_process():
        # text = "Las niñas juegan en los Estados Unidos. El Tío Sam observa a los niños. " \
        #        "Yo bajo con el hombre bajo a tocar el bajo bajo la escalera. " \
        #        "Yo bajo el volumen de los niños."

        text = ["Los niños juegan en los Estados Unidos. El Tío Sam observa a los niños.",
                " Yo bajo con el hombre bajo a tocar el bajo bajo la escalera.",
                " Yo bajo el volumen de los niños."]

        conf = Configuration()
        nlp = SpacyModel.getInstance().model
        conf.load_dict()
        process = PreProcess(conf, nlp)

        # process._pre_process()
        process.pre_process_batches()
        process.create_full_dataset_vectorizer()
