import json

import requests


class ITest:

    @staticmethod
    def local_server_up(server_url):
        response = requests.get(server_url)
        return response.status_code == 200

    @staticmethod
    def do_request(server_url, source, data_path, model='', file_path='', lang='es'):
        body = {
            "source": source,
            "data": data_path,
            "dictionary": "nlp/resources/dic_es.txt",
            "model": model,
            "file": file_path,
            "lang": lang
        }
        payload = json.dumps(body)
        header = {"Content-Type": "application/json"}
        response = requests.post(server_url, data=payload, headers=header)
        if response.status_code != 200:
            return None

        return response
