import inspect
import os
import sys

import boto3
# start - JAEGER
import opentracing
from flask import request, jsonify
from flask_restplus import Resource, fields
from jaeger_client import Config
from opentracing_utils import trace_requests

from commonsLib import loggerElk

# end - JAEGER

GLOBAL_DEBUG = True


class GbcMachineLearningService(Resource):
    from api import api

    logger = loggerElk(__name__, True)
    nlp = None

    machineLearningService = api.model('MachineLearningService', {
        'source': fields.String(required=True, description='Source of the data (PLAINTEXT | FILE | IMAGE | S3)'),
        'data': fields.String(required=True, description='Repo resource identifier (url)'),
        'domain': fields.String(required=True, description='Repo resource identifier (domain name)'),
        'model': fields.String(required=True, description='Source of classifier '
                                                          '(BAGGING | BOOSTING_ADA | BOOSTING_SGD '
                                                          '| DECISION_TREE | EXTRA_TREES | NAIVE_BAYES_MULTI '
                                                          '| NAIVE_BAYES_COMPLEMENT | RANDOM_FOREST | VOTING '
                                                          '| CNN_NETWORK)'),
        'lang': fields.String(required=True, description='Language (es, en)'),
    })

    def __init__(self, *args, **kwargs):
        # start - JAEGER
        config = Config(config={'sampler': {'type': 'const', 'param': 1},
                                'logging': True
                                },
                        service_name=__name__)
        config.initialize_tracer()
        super().__init__(*args, **kwargs)

    trace_requests()  # noqa

    # end - JAEGER
    @api.doc(
        description='Machine Learning Service',
        responses={
            200: 'OK',
            400: 'Invalid Argument',
            500: 'Internal Error'})
    @api.expect(machineLearningService)
    def post(self):
        root_span = None
        try:
            self.logger.Information('GbcMachineLearningService::POST - init')
            # start - JAEGER
            root_span = opentracing.tracer.start_span(operation_name=inspect.currentframe().f_code.co_name)
            # end - JAEGER

            request_payload = request.get_json()
            # self.logger.LogInput('GbcMachineLearningService::POST: ', request_payload)

            source = request_payload['source']
            data = request_payload['data']
            domain = request_payload['domain']
            model = request_payload['model']
            lang = request_payload['lang']

            if source == 'PLAINTEXT':
                response = 'PLAINTEXT'
            elif source == 'FILE':
                response = 'FILE'
            elif source == 'IMAGE':
                response = 'IMAGE'
            elif source == 'S3':
                s3 = self.getS3Session()
                return {'result': 'not implemented'}
            else:
                response = ''
                raise Exception('No valid source provided')

            res = {
                'result': 'ok',
                'response': response
            }

            return jsonify(res)

        except Exception as e:
            self.logger.Error('ERROR - GbcMachineLearningService::POST' + str(e.args), sys.exc_info())
            return {'message': 'Something went wrong: ' + str(e)}, 500

        finally:
            root_span.finish()

    class Student(object):
        def __init__(self, first_name: str, last_name: str):
            self.first_name = first_name
            self.last_name = last_name

    @classmethod
    def getS3Session(cls):
        session = boto3.Session(
            aws_access_key_id=os.environ['ENV_ACCESS_KEY_ID'],
            aws_secret_access_key=os.environ['ENV_SECRET_ACCESS_KEY']
        )
        s3 = session.client(u's3')
        return s3
