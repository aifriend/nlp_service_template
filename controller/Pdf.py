import os
import re

from pdf2image import convert_from_path
from tika import parser


class Pdf:

    @staticmethod
    def to_image(pdf_list):
        for pdf in pdf_list:
            pages = convert_from_path(pdf, 500)
            filename, file_extension = os.path.splitext(pdf)
            pages[0].save(filename + '.jpg')

    def to_text(self, pdf_list):
        count = 0
        for pdf in pdf_list:
            # Parse data from file
            file_data = parser.from_file(pdf)
            # Get files text content
            text = file_data['content']

            print(pdf)
            text = self.clean_text(text)
            print(text)

            filename, file_extension = os.path.splitext(pdf)
            if text is not None and len(text) > 50:
                new_file = open(filename + '.txt', mode="w", encoding="utf-8")
                new_file.write(text)
                new_file.flush()
                new_file.close()

            count += 1

        print(count, " processed!")

    @staticmethod
    def clean_text(text):
        if text is None:
            return ''

        new_ = re.sub(r'\n\n', r'\n', text)
        new_ = re.sub(r'\t', r' ', new_)
        new_ = re.sub(r' {2}', r' ', new_)
        new_ = re.sub(r'--', r'-', new_)

        return new_.strip()
