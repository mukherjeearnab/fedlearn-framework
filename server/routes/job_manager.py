'''
Client Management Routing Module
'''

from flask import Blueprint, jsonify, request, send_file
from helpers.semaphore import Semaphore
from helpers.logging import logger
from helpers.kvstore import kv_delete
from apps.job_loader import load_job, start_job

ROUTE_NAME = 'job-manager'
blueprint = Blueprint(ROUTE_NAME, __name__)

STATE_LOCK = Semaphore()

JOBS = {}
CONFIGS = {}
# JOB_THREADS = {}


@blueprint.route('/')
def root():
    '''
    blueprint root route, provides a brief description of the route,
    with some additional information.
    '''
    data = {'message': f'This is the \'{ROUTE_NAME}\' router.'}
    return jsonify(data)


@blueprint.route('/load')
def load_job_route():
    '''
    init route, called by server to initialize the job into the state machine
    '''
    job_id = request.args['job_id']

    # STATE_LOCK.acquire()
    try:
        load_job(job_id, CONFIGS)
    except Exception as e:
        logger.error(f'Failed to Load Job Instance {e}')

    # STATE_LOCK.release()

    return jsonify({'message': 'Job instance loaded successfully!'})


@blueprint.route('/delete')
def delete_job_route():
    '''
    delete route, called by server to delete the job from the state machine
    '''
    job_id = request.args['job_id']

    # STATE_LOCK.acquire()
    try:
        # if job is terminated, only then it can be deleted
        if JOBS[job_id].job_status['process_phase'] == 3:
            kv_delete(job_id)
            del JOBS[job_id]
            del CONFIGS[job_id]
            logger.info(f'Job [{job_id}] deleted successfully!')
        else:
            logger.error(
                f'Failed to Delete Job Instance {job_id}. Reason: Job Is not Terminated.')
            logger.info(f'Please Wait for Job [{job_id}] to terminate.')
    except Exception as e:
        logger.error(f'Failed to Delete Job Instance. {e}')

    # STATE_LOCK.release()

    return jsonify({'message': 'Job instance deleted successfully!'})


@blueprint.route('/start')
def start_job_route():
    '''
    init route, called by server to initialize the job into the state machine
    '''
    job_id = request.args['job_id']

    # STATE_LOCK.acquire()
    try:
        start_job(job_id, CONFIGS, JOBS)
    except Exception as e:
        logger.error(f'Failed to Load Job Instance {e}')

    job_state = JOBS[job_id].get_state()
    # JOB_THREADS[job_id] = resp

    # STATE_LOCK.release()

    return jsonify(job_state)


@blueprint.route('/list')
def list_jobs():
    '''
    get the list of all the jobs in the state machine
    '''
    # STATE_LOCK.wait()
    jobs = list(JOBS.keys())

    return jsonify(jobs)


@blueprint.route('/get')
def get():
    '''
    GET route, get all the state of a given job
    '''
    job_id = request.args['job_id']

    # STATE_LOCK.wait()
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

    # STATE_LOCK.acquire()

    if job_id in JOBS.keys():
        JOBS[job_id].update_client_status(client_id, client_status)
    else:
        status = 404
    # STATE_LOCK.release()

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

    # STATE_LOCK.acquire()

    if job_id in JOBS:
        JOBS[job_id].append_client_params(client_id, client_params)
    else:
        status = 404
    # STATE_LOCK.release()

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
            CHUNK_DIR_NAME += f'-{chunk}'

        DATASET_CHUNK_PATH = f"../datasets/deploy/{JOBS[job_id].dataset_params['prep']['file']}/chunks/{CHUNK_DIR_NAME}"

        chunk_id = 0
        for i, client in enumerate(JOBS[job_id].exec_params['client_info']):
            if client['client_id'] == client_id:
                chunk_id = i

        file_name = f'{chunk_id}.tuple'
        file_path = f'{DATASET_CHUNK_PATH}/{file_name}'

        return send_file(file_path, mimetype='application/octet-stream',
                         download_name=file_name, as_attachment=True)
    else:
        status = 404
        return jsonify({'message': f'Dataset File for Client [{client_id}] not found for Job [{job_id}]!'}), status
