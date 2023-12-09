'''
The Job Management API module
'''

import os
import argparse
from helpers.http import get, post
from dotenv import load_dotenv

load_dotenv()

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--port")
args = parser.parse_args()

SERVER_PORT = int(args.port if args.port else os.getenv('SERVER_PORT'))

SERVER_URL = f'http://localhost:{SERVER_PORT}'


def load_job(job_config: str, job_name: str):
    '''
    Call the Job Load Route and load the job
    '''
    get(f'{SERVER_URL}/job_manager/load',
        {'job_id': job_name, 'job_config': job_config})


def delete_job(job_name: str):
    '''
    Call the Job Delete Route and delete the job
    '''
    get(f'{SERVER_URL}/job_manager/delete', {'job_id': job_name})


def start_job(job_name: str):
    '''
    Call the Job Start Route and start the job
    '''
    get(f'{SERVER_URL}/job_manager/start', {'job_id': job_name})


def get_job(job_name: str, params=False):
    '''
    Call get Job Configuration
    '''
    if params:
        return get(f'{SERVER_URL}/job_manager/get_params', {'job_id': job_name})
    else:
        return get(f'{SERVER_URL}/job_manager/get_exec', {'job_id': job_name})


################################################################
# Job Specific Routes
################################################################


def allow_start_training(job_name: str):
    '''
    Call to Allow Start Training
    '''
    post(f'{SERVER_URL}/job_manager/allow_start_training', {'job_id': job_name})


def terminate_training(job_name: str):
    '''
    Call to Terminate Training
    '''
    post(f'{SERVER_URL}/job_manager/terminate_training', {'job_id': job_name})


def set_central_model_params(job_name: str, params: dict):
    '''
    Call to Set Cental Model Params
    '''
    post(f'{SERVER_URL}/job_manager/set_central_model_params',
         {'job_id': job_name, 'central_params': params})


def set_abort(job_name: str):
    '''
    Call to Abort Job
    '''
    post(f'{SERVER_URL}/job_manager/set_abort', {'job_id': job_name})
