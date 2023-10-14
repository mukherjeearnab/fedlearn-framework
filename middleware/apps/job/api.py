'''
The Job Management API module
'''

# import os
from helpers.http import get, post
from global_kvset import app_globals


def load_job(job_name: str, job_manifest: dict):
    '''
    Call the Job Load Route and load the job
    '''
    res = post(f'{app_globals.get("LOOPBACK_URL")}/job_manager/load',
               {'job_id': job_name, 'job_manifest': job_manifest})
    return res['exec_status']


def delete_job(job_name: str):
    '''
    Call the Job Delete Route and delete the job
    '''
    get(f'{app_globals.get("LOOPBACK_URL")}/job_manager/delete',
        {'job_id': job_name})


def start_job(job_name: str):
    '''
    Call the Job Start Route and start the job
    '''
    get(f'{app_globals.get("LOOPBACK_URL")}/job_manager/start', {'job_id': job_name})


def get_job(job_name: str, params=False):
    '''
    Call get Job Configuration
    '''
    if params:
        return get(f'{app_globals.get("LOOPBACK_URL")}/job_manager/get_params', {'job_id': job_name})
    else:
        return get(f'{app_globals.get("LOOPBACK_URL")}/job_manager/get_exec', {'job_id': job_name})


################################################################
# Job Specific Routes
################################################################

def allow_dataset_download(job_name: str):
    '''
    Call to Allow Start Training
    '''
    post(f'{app_globals.get("LOOPBACK_URL")}/job_manager/allow_dataset_download',
         {'job_id': job_name})


def allow_start_training(job_name: str):
    '''
    Call to Allow Start Training
    '''
    post(f'{app_globals.get("LOOPBACK_URL")}/job_manager/allow_start_training',
         {'job_id': job_name})


def terminate_training(job_name: str):
    '''
    Call to Terminate Training
    '''
    post(f'{app_globals.get("LOOPBACK_URL")}/job_manager/terminate_training',
         {'job_id': job_name})


def set_central_model_params(job_name: str, params: dict):
    '''
    Call to Set Cental Model Params
    '''
    post(f'{app_globals.get("LOOPBACK_URL")}/job_manager/set_central_model_params',
         {'job_id': job_name, 'central_params': params})


def set_abort(job_name: str):
    '''
    Call to Abort Job
    '''
    post(f'{app_globals.get("LOOPBACK_URL")}/job_manager/set_abort',
         {'job_id': job_name})
