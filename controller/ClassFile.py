import imghdr
import os
import pickle
from pathlib import PurePath

import numpy as np
from binaryornot.check import is_binary


class ClassFile:

    @staticmethod
    def has_text_file(path):
        try:
            file_test = None
            if os.path.isdir(path):
                file_test = ClassFile.list_files(path).pop()
            elif os.path.isfile(path):
                file_test = path
            if file_test is not None:
                return not is_binary(file_test)
            return False
        except Exception:
            return False

    @staticmethod
    def has_media_file(path):
        try:
            file_test = None
            if os.path.isdir(path):
                file_test = ClassFile.list_files(path).pop()
            elif os.path.isfile(path):
                file_test = path
            if file_test is not None:
                return is_binary(file_test) and imghdr.what(file_test) in ["jpg", "jpeg", "png"]
            return False
        except Exception:
            return False

    @staticmethod
    def list_directory(path):
        """
        list all directories under specific route
        """
        files = []
        with os.scandir(path) as entries:
            for entry in entries:
                files.append(entry.name)

        return files

    @staticmethod
    def list_files(path):
        """
        list all files under specific route
        """
        files = []
        # r=root, d=directories, f = files
        for r, d, f in os.walk(path):
            for file in f:
                files.append(os.path.join(r, file))

        return files

    @staticmethod
    def list_pdf_files(path):
        """
        list all pdf files under specific route
        """
        files = []
        # r=root, d=directories, f = files
        for r, d, f in os.walk(path):
            for file in f:
                if '.pdf' in file:
                    files.append(os.path.join(r, file))

        return files

    @staticmethod
    def list_files_ext(path, ext):
        """
        list all files under path with given extension
        """
        files = []
        # r=root, d=directories, f = files
        for r, d, f in os.walk(path):
            for file in f:
                if ext.lower() in file.lower():
                    files.append(os.path.join(r, file))

        return files

    @staticmethod
    def get_file_ext(file):
        """
        get the extension of a file
        """
        return os.path.splitext(file)[1]

    @staticmethod
    def get_dir_name(file):
        """
        get the the full path dir container of the file
        """
        return os.path.dirname(file)

    @staticmethod
    def get_containing_dir_name(file):
        """
        get just the name of the containing dir of the file
        """
        return PurePath(file).parent.name

    @staticmethod
    def file_base_name(file_name):
        """
        get the path and name of the file without the extension
        """
        if '.' in file_name:
            separator_index = file_name.index('.')
            base_name = file_name[:separator_index]
            return base_name
        else:
            return file_name

    def get_file_name(self, path):
        """
        get the name of the file without the extension
        """
        file_name = os.path.basename(path)
        return self.file_base_name(file_name)

    @staticmethod
    def create_dir(directory):
        """
        create a directory
        """
        if not os.path.exists(directory):
            os.makedirs(directory)

    @staticmethod
    def list_to_file(set_, file_):
        """
        save a list to a file using pickle dump
        """
        with open(file_, 'wb') as fp:
            pickle.dump(sorted(list(set_)), fp)

    @staticmethod
    def to_txtfile(data, file_):
        """
        save data as a text file
        """
        with open(file_, "w") as output:
            output.write(str(data))

    @staticmethod
    def file_to_list(file_, binary=True):
        """
        read a list from a pickle file
        """
        list_ = []
        if os.path.getsize(file_) > 0:
            if binary:
                with open(file_, 'rb') as fp:
                    list_ = pickle.load(fp)
            else:
                with open(file_, 'r') as fp:
                    list_ = fp.readlines()
        return sorted(list(list_))

    @staticmethod
    def csv_to_numpy_image(csv_file):
        """
        load numpy image from csv file
        """
        np.loadtxt(csv_file)

    @staticmethod
    def get_text(filename):
        """
        read from file as text
        """
        f = open(filename, "r", encoding="ISO-8859-1")
        return f.read()

    @staticmethod
    def save_sparse_csr(filename, mat):
        """
        save a sparse vector as a list of its elements
        """
        listing = mat.toarray().tofile(filename, sep=",", format="%f")

    @staticmethod
    def save_model(filename, model):
        """
        save scikit-learn model
        """
        pickle.dump(model, open(filename, 'wb'))

    @staticmethod
    def load_model(filename):
        """
        load scikit-learn model
        """
        if os.path.getsize(filename) > 0:
            with open(filename, 'rb') as fp:
                return pickle.load(fp)
