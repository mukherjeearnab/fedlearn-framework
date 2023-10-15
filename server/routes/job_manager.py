'''
Client Management Routing Module
'''
import threading
from flask import Blueprint, jsonify, request, send_file
# from helpers.semaphore import Semaphore
from helpers.logging import logger
from helpers.kvstore import kv_delete
from apps.job.management.loader import load_job
from apps.job.management.starter import start_job
from apps.client.api import delete_job_at_middlewares

ROUTE_NAME = 'job-manager'
blueprint = Blueprint(ROUTE_NAME, __name__)

STATE_LOCK = threading.Lock()

JOBS = {}
CONFIGS = {}


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

    STATE_LOCK.acquire()
    # try:
    load_job(job_id, CONFIGS)
    # except Exception as e:
    #     logger.error(f'Failed to Load Job Instance {e}')

    STATE_LOCK.release()

    return jsonify({'message': 'Job instance loaded successfully!'})


@blueprint.route('/delete')
def delete_job_route():
    '''
    delete route, called by server to delete the job from the state machine
    '''
    job_id = request.args['job_id']

    STATE_LOCK.acquire()
    # try:
    # if job is terminated, only then it can be deleted
    if JOBS[job_id][0].job_status['process_phase'] == 3:
        for client in JOBS[job_id][0].job_status['client_info']:
            if client['is_middleware']:
                delete_job_at_middlewares(client['client_id'], job_id)

        kv_delete(job_id)
        del JOBS[job_id]
        del CONFIGS[job_id]
        logger.info(f'Job [{job_id}] deleted successfully!')
    else:
        logger.error(
            f'Failed to Delete Job Instance {job_id}. Reason: Job Is not Terminated.')
        logger.info(f'Please Wait for Job [{job_id}] to terminate.')
    # except Exception as e:
        # logger.error(f'Failed to Delete Job Instance. {e}')

    STATE_LOCK.release()

    return jsonify({'message': 'Job instance deleted successfully!'})


@blueprint.route('/start')
def start_job_route():
    '''
    init route, called by server to initialize the job into the state machine
    '''
    job_id = request.args['job_id']

    # try:
    start_job(job_id, CONFIGS, JOBS)
    job_state = JOBS[job_id][0].get_state()
    # except Exception as e:
    #     logger.error(f'Failed to Start Job Instance {e}')
    #     job_state = {'message': 'Job instance not found'}

    return jsonify(job_state)


@blueprint.route('/list')
def list_jobs():
    '''
    get the list of all the jobs in the state machine
    '''

    jobs = list(JOBS.keys())

    return jsonify(jobs)


@blueprint.route('/get')
def get():
    '''
    GET route, get all the state of a given job
    '''
    job_id = request.args['job_id']

    job_state = JOBS[job_id][0].get_state()
    job_state['exec_params'] = JOBS[job_id][1].get_state()['exec_params']

    return jsonify(job_state)


@blueprint.route('/get_exec')
def get_exec():
    '''
    GET route, get all the state of a given job
    '''
    job_id = request.args['job_id']

    job_state = JOBS[job_id][0].get_state()

    return jsonify(job_state)


@blueprint.route('/get_params')
def get_params():
    '''
    GET route, get all the state of a given job
    '''
    job_id = request.args['job_id']

    job_state = JOBS[job_id][1].get_state()

    return jsonify(job_state)


################################################################
# Job Specific Routes
################################################################


@blueprint.route('/allow_start_training', methods=['POST'])
def allow_start_training():
    '''
    ROUTE to update the client status
    '''
    payload = request.get_json()
    status = 200

    job_id = payload['job_id']

    STATE_LOCK.acquire()

    if job_id in JOBS.keys():
        JOBS[job_id][0].allow_start_training()
    else:
        status = 404
    STATE_LOCK.release()

    return jsonify({'message': 'Training Allowed!' if status == 200 else 'Training NOT Allowed!'}), status


@blueprint.route('/terminate_training', methods=['POST'])
def terminate_training():
    '''
    ROUTE to terminate the job
    '''
    payload = request.get_json()
    status = 200

    job_id = payload['job_id']

    STATE_LOCK.acquire()

    if job_id in JOBS.keys():
        JOBS[job_id][0].terminate_training()
    else:
        status = 404
    STATE_LOCK.release()

    return jsonify({'message': 'Training Terminated!' if status == 200 else 'Training NOT Terminated!'}), status


@blueprint.route('/set_abort', methods=['POST'])
def set_abort():
    '''
    ROUTE to Abort job
    '''
    payload = request.get_json()
    status = 200

    job_id = payload['job_id']

    STATE_LOCK.acquire()

    if job_id in JOBS.keys():
        JOBS[job_id][0].set_abort()
    else:
        status = 404
    STATE_LOCK.release()

    return jsonify({'message': 'Training Aborted!' if status == 200 else 'Training NOT Aborted!'}), status


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
        JOBS[job_id][0].update_client_status(client_id, client_status)
    else:
        status = 404
    STATE_LOCK.release()

    return jsonify({'message': 'Status updated!' if status == 200 else 'Update failed!'}), status


@blueprint.route('/set_central_model_params', methods=['POST'])
def set_central_model_params():
    '''
    ROUTE to update the client status
    '''
    payload = request.get_json()
    status = 200

    job_id = payload['job_id']
    central_params = payload['central_params']

    STATE_LOCK.acquire()

    if job_id in JOBS.keys():
        JOBS[job_id][1].set_central_model_params(central_params)
    else:
        status = 404
    STATE_LOCK.release()

    return jsonify({'message': 'Central Params SET!' if status == 200 else 'Central Params NOT SET!'}), status


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
        JOBS[job_id][1].append_client_params(client_id, client_params)
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

        if JOBS[job_id][0].hierarchical:
            client_split_key = 'splits'
        else:
            client_split_key = 'clients'

        for chunk in JOBS[job_id][0].client_params['dataset']['distribution'][client_split_key]:
            CHUNK_DIR_NAME += f'-{chunk}'

        DATASET_PREP_MOD = JOBS[job_id][0].dataset_params['prep']['file']
        DATASET_DIST_MOD = JOBS[job_id][0].client_params['dataset']['distribution']['distributor']['file']
        DATASET_CHUNK_PATH = f"../../datasets/deploy/{DATASET_PREP_MOD}/chunks/{DATASET_DIST_MOD}/{CHUNK_DIR_NAME}"

        chunk_id = 0
        for i, client in enumerate(JOBS[job_id][0].job_status['client_info']):
            if client['client_id'] == client_id:
                chunk_id = i

        file_name = f'{chunk_id}.tuple'
        file_path = f'{DATASET_CHUNK_PATH}/{file_name}'

        return send_file(file_path, mimetype='application/octet-stream',
                         download_name=file_name, as_attachment=True)
    else:
        status = 404
        return jsonify({'message': f'Dataset File for Client [{client_id}] not found for Job [{job_id}]!'}), status
