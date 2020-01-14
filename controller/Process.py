from PIL import Image

from controller.ClassFile import ClassFile
from controller.ClassImage import ClassImage
from controller.Pdf import Pdf
from model.Document import Document


class Process:

    def __init__(self, conf):
        self.conf = conf
        self.from_file = ClassFile()
        self.from_pdf = Pdf()
        self.from_image = ClassImage()

    def create_examples(self, process_pdf=False):
        path = self.conf.path

        if process_pdf:
            pdf_list = self.from_file.list_files_ext(path, '.pdf')
            self.from_pdf.to_image(pdf_list)

        i_list = self.from_file.list_files_ext(path, '.jpg')
        print(i_list)

        docs = []
        categories = set()

        for image in i_list:
            cropped = self.from_image.crop_image_loaded(
                self.from_image.resize_image_loaded(
                    self.from_image.load_image(image),
                    self.conf.resize_width,
                    self.conf.resize_height), self.conf.crop_width, self.conf.crop_height)
            name = self.from_file.get_file_name(image)
            categories.add(name[4:-3])
            ext = self.from_file.get_file_ext(image)
            file = self.from_file.get_dir_name(image) + self.conf.sep + name + '_crop' + ext
            # Image.fromarray(cropped).save(file)
            examples = self.from_image.generate_examples(cropped, self.conf.examples_per_case)
            i = 0
            directory = self.conf.examples_dir + self.conf.sep + name[4:-3]
            self.from_file.create_dir(directory)
            for example in examples:
                Image.fromarray(example).save(directory + self.conf.sep + name[4:] + '_' + str(i).zfill(3) + ext)
                docs.append(Document(example, name[4:-3]))
                i += 1

        self.from_file.list_to_file(categories, self.conf.examples_dir + self.conf.sep + self.conf.cat_file)

    def create_svh_data(self, process_pdf=False):
        path = self.conf.working_path

        if process_pdf:
            pdf_list = self.from_file.list_files_ext(path, '.pdf')
            self.from_pdf.to_text(pdf_list)
