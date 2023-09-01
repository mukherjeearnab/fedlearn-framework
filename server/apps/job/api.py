'''
The Job Management API module
'''

import os
from helpers.http import get, post
from dotenv import load_dotenv

load_dotenv()

SERVER_URL = f'http://localhost:{int(os.getenv("SERVER_PORT"))}'


def load_job(job_name: str):
    '''
    Call the Job Load Route and load the job
    '''
    get(f'{SERVER_URL}/job_manager/load', {'job_id': job_name})


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


def get_job(job_name: str):
    '''
    Call get Job Configuration
    '''
    return get(f'{SERVER_URL}/job_manager/get', {'job_id': job_name})


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