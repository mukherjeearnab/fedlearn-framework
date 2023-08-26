'''
The Job Management API module
'''

import os
from helpers.http import get
from dotenv import load_dotenv

load_dotenv()

SERVER_URL = f'http://localhost:{int(os.getenv("SERVER_PORT"))}'


def load_job(job_name: str):
    '''
    Call the Job Load Route and load the job
    '''
    get(f'{SERVER_URL}/job_manager/load', {'job_id': job_name})


def start_job(job_name: str):
    '''
    Call the Job Start Route and start the job
    '''
    get(f'{SERVER_URL}/job_manager/start', {'job_id': job_name})
