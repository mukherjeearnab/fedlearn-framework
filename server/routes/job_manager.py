'''
Client Management Routing Module
'''

from flask import Blueprint, jsonify, request
from helpers.semaphore import Semaphore
from helpers.logging import logger
from apps.training_job import TrainingJobManager


ROUTE_NAME = 'job-manager'
blueprint = Blueprint(ROUTE_NAME, __name__)

STATE_LOCK = Semaphore()

JOBS = dict()


@blueprint.route('/')
def root():
    '''
    blueprint root route, provides a brief description of the route,
    with some additional information.
    '''
    data = {'message': f'This is the \'{ROUTE_NAME}\' router.'}
    return jsonify(data)


@blueprint.route('/init')
def init():
    '''
    init route, called by server to initialize the job into the state machine
    '''
    job_name = request.args['job']

    STATE_LOCK.acquire()
    try:
        JOBS[job_name] = TrainingJobManager(project_name=job_name,
                                            client_params=dict(),
                                            server_params=dict(),
                                            dataset_params=dict(),
                                            load_from_db=True)
        logger.info(f'Created Job Instance for Job {job_name}.')
    except Exception as e:
        logger.error(f'Failed to Retrieve Job Instance')

    job_state = JOBS[job_name].get_state()
    STATE_LOCK.release()

    return jsonify(job_state)


@blueprint.route('/list')
def list():
    '''
    get the list of all the jobs in the state machine
    '''
    STATE_LOCK.wait()
    jobs = [job_name for job_name in JOBS.keys()]

    return jsonify(jobs)


@blueprint.route('/get')
def get():
    '''
    GET route, get all the state of a given job
    '''
    job_name = request.args['job']

    STATE_LOCK.wait()
    job_state = JOBS[job_name].get_state()

    return jsonify(job_state)


@blueprint.route('/update_client_status', methods=['POST'])
def update_client_status():
    '''
    ROUTE to update the client status
    '''
    payload = request.get_json()
    status = 200

    client_id = payload['client_id']
    client_status = payload['client_status']
    job_name = payload['job_name']

    STATE_LOCK.acquire()

    if job_name in JOBS.keys():
        JOBS[job_name].update_client_status(client_id, client_status)
    else:
        status = 404
    STATE_LOCK.release()

    return jsonify({'message': 'Status updated!' if status == 200 else 'Update failed!'}), status


@blueprint.route('/append_client_params', methods=['POST'])
def append_client_params():
    '''
    ROUTE to update the client status
    '''
    payload = request.get_json()
    status = 200

    client_id = payload['client_id']
    client_params = payload['client_params']
    job_name = payload['job_name']

    STATE_LOCK.acquire()

    if job_name in JOBS.keys():
        JOBS[job_name].append_client_params(client_id, client_params)
    else:
        status = 404
    STATE_LOCK.release()

    return jsonify({'message': 'Params Added!' if status == 200 else 'Method failed!'}), status
