'''
Job Getters Module
'''
from multiprocessing import Process
from helpers.logging import logger
from helpers.http import get
from apps.job.process import job_process


def get_jobs_from_server(middleware_id: str, jobs_registry: dict, server_url: str):
    '''
    Job Checker method
    '''

    url = f'{server_url}/job_manager/list'

    # logger.info(f'Fetching Job list from Server at {url}')

    jobs = get(url, {})

    # remove deleted jobs
    for job_id in jobs_registry['job_ids']:
        if job_id not in jobs:
            # remove job ID from jobs registry.
            jobs_registry['job_ids'].remove(job_id)

    # add newly added jobs
    for job_id in jobs:
        if job_id not in jobs_registry['job_ids']:
            # add job ID to jobs registry.
            jobs_registry['job_ids'].append(job_id)

            # start job process
            job_proc = get_job_manifest(middleware_id, job_id, server_url)

            # add job process to registry
            jobs_registry['jobs'][job_id]['process'] = job_proc


def get_job_manifest(middleware_id: str, job_id: str, server_url: str):
    '''
    Download job manifest from server for [job_id]
    '''

    url = f'{server_url}/job_manager/get'

    logger.info(f'Fetching Job Manifest for [{job_id}] from Server at {url}')

    manifest = get(url, {'job_id': job_id})

    for middleware in manifest['job_status']['client_info']:
        if middleware['client_id'] == middleware_id:
            logger.info(f'Starting Job Process for Job [{job_id}]')

            # start new job thread
            job_proc = Process(target=job_process,
                               args=(middleware_id, job_id, manifest, server_url), name=f'job_{job_id}')

            # start job process
            job_proc.start()

            # return the job process
            return job_proc

    return None
