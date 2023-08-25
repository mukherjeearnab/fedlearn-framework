'''
Client Management Routing Module
'''

from flask import Blueprint, jsonify, request, send_file
from helpers.semaphore import Semaphore
from helpers.logging import logger
from apps.training_job import TrainingJobManager


ROUTE_NAME = 'job-manager'
blueprint = Blueprint(ROUTE_NAME, __name__)

STATE_LOCK = Semaphore()

JOBS = {}


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
    job_id = request.args['job_id']

    STATE_LOCK.acquire()
    try:
        JOBS[job_id] = TrainingJobManager(project_name=job_id,
                                          client_params={},
                                          server_params={},
                                          dataset_params={},
                                          load_from_db=True)
        logger.info(f'Created Job Instance for Job {job_id}.')
    except Exception as e:
        logger.error(f'Failed to Retrieve Job Instance {e}')

    job_state = JOBS[job_id].get_state()
    STATE_LOCK.release()

    return jsonify(job_state)


@blueprint.route('/list')
def list_jobs():
    '''
    get the list of all the jobs in the state machine
    '''
    STATE_LOCK.wait()
    jobs = list(JOBS.keys())

    return jsonify(jobs)


@blueprint.route('/get')
def get():
    '''
    GET route, get all the state of a given job
    '''
    job_id = request.args['job_id']

    STATE_LOCK.wait()
    job_state = JOBS[job_id].get_state()

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
    job_id = payload['job_id']

    STATE_LOCK.acquire()

    if job_id in JOBS.keys():
        JOBS[job_id].update_client_status(client_id, client_status)
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
    job_id = payload['job_id']

    STATE_LOCK.acquire()

    if job_id in JOBS:
        JOBS[job_id].append_client_params(client_id, client_params)
    else:
        status = 404
    STATE_LOCK.release()

    return jsonify({'message': 'Params Added!' if status == 200 else 'Method failed!'}), status


@blueprint.route('/download_dataset')
def download_dataset():
    '''
    ROUTE to download dataset, based on client_id and job_name
    '''
    client_id = request.args['client_id']
    job_id = request.args['job_id']
    status = 200

    if job_id in JOBS:
        CHUNK_DIR_NAME = 'dist'
        for chunk in JOBS[job_id].client_params['dataset']['distribution']['clients']:
            CHUNK_DIR_NAME.join(f'-{chunk}')

        DATASET_CHUNK_PATH = f"../datasets/deploy/{JOBS[job_id].dataset_params['prep']['file']}/chunks/{CHUNK_DIR_NAME}"

        file_name = f'{client_id}.tuple'
        file_path = f'{DATASET_CHUNK_PATH}/{file_name}'

        return send_file(file_path, mimetype='application/octet-stream',
                         download_name=file_name, as_attachment=True)
    else:
        status = 404
        return jsonify({'message': f'Dataset File for Client [{client_id}] not found for Job [{job_id}]!'}), status
