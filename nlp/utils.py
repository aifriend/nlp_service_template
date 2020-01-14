import re
import urllib
import urllib.request

from PyDictionary import PyDictionary
from bs4 import BeautifulSoup as soup

from commonsLib import loggerElk

logger = loggerElk(__name__, True)


def clean_text(text):
    if text is None:
        return ''

    new_ = re.sub('\n\n', '\n', text)
    new_ = re.sub('\t', ' ', new_)
    new_ = re.sub(' {2}', ' ', new_)
    new_ = re.sub('--', '-', new_)
    new_ = re.sub('\|', '', new_)

    new_ = re.sub('á', 'a', new_)
    new_ = re.sub('é', 'e', new_)
    new_ = re.sub('í', 'i', new_)
    new_ = re.sub('ó', 'o', new_)
    new_ = re.sub('ú', 'u', new_)
    new_ = re.sub('Á', 'A', new_)
    new_ = re.sub('É', 'E', new_)
    new_ = re.sub('Í', 'I', new_)
    new_ = re.sub('Ó', 'O', new_)
    new_ = re.sub('Ú', 'U', new_)

    return new_.strip()


def synonym(word):
    try:
        s = ''
        with urllib.request.urlopen('https://educalingo.com/en/dic-es/{}'.format(word)) as url:
            s = url.read()
        final_results = re.findall('\w+', [i.text for i in
                                           soup(s, 'html.parser')
                                   .find_all('div', {"class": 'contenido_sinonimos_antonimos'})][0])
        print(final_results)
        return final_results
    except Exception as exc:
        logger.Error("Error loading model/config.yml...")
        return None


def checker(word):
    dictionary = PyDictionary()
    print(dictionary.meaning(word))


def split_by_size(text, max_size=10):
    data = text.split(' ')
    full_text = []
    buffer = ''
    for element in data:
        if len(buffer) + len(element) < max_size:
            buffer = buffer + ' ' + element
        else:
            if buffer is not '':
                full_text.append(buffer.strip())
            buffer = element

    if buffer is not '':
        full_text.append(buffer.strip())

    print(full_text)
    return full_text
