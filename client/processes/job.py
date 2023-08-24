'''
Job Process Management module
'''
from helpers.logging import logger
from helpers.http import get, download_file
from helpers.client_status import update_client_status
from helpers.server_listeners import listen_to_dataset_download_flag, listen_to_start_training
from processes.training import data_preprocessing


def job_process(client_id: str, job_id: str, job_manifest: dict, server_url: str):
    '''
    The Job Process method
    1. ACK of job manifest to server, and update client status to 1.
    2. Listen to download_dataset to turn true, then download dataset.
    3. Preprocess dataset.
    4. ACK of dataset to server, and update client status to 2.
    5. Listen to check when process phase turns 1.
    6. Download global parameters from server.
    7. ACK of global parameters to server, and update client status to 3.
    8. Perform local training.
    9. Send back locally trained model parameters and ACK of model update, and update client status to 4.
    10. Listen to check when process phase change to 2.
    11. Listen to check when process phase change to 1 or 3.
    12. If process phase is 1, repeat steps 6-11, else terminate process.
    '''

    # Step 1: ACK of job manifest to server, and update client status to 1.
    update_client_status(client_id, job_id, 1, server_url)

    # Step 2: Listen to download_dataset to turn true, then download dataset.
    # 2.1 listen to download dataset flag
    listen_to_dataset_download_flag(job_id, server_url)

    # 2.2 download dataset to ./datasets/[job_id]/dataset.tuple
    download_file(f'{server_url}/job_manager/download_dataset?job_id={job_id}&client_id={client_id}',
                  f'./datasets/{job_id}/dataset.tuple')

    # Step 3: Preprocess dataset.
    (train_set, test_set) = data_preprocessing(f'./datasets/{job_id}',
                                               job_manifest['client_params']['dataset']['preprocessor']['content'],
                                               list(job_manifest['client_params']['train_test_split'].values()))

    # Step 4: ACK of dataset to server, and update client status to 2.
    update_client_status(client_id, job_id, 2, server_url)

    # Step 5: Listen to check when process phase turns 1.
    listen_to_start_training(job_id, server_url)

    # some logging vars
    global_round = 1

    # round loop for steps 6-12
    while True:
        # Step 6: Download global parameters from server.

        # Step 7: ACK of global parameters to server, and update client status to 3.
        update_client_status(client_id, job_id, 3, server_url)

        # Step 8 Perform local training.

        # Step 9 Send back locally trained model parameters
        # this will update client status to 4 on the server automatically.

        # Step 10: Listen to check when process phase change to 2.

        # Step 11: Listen to check when process phase change to 1 or 3.

        # update round count
        global_round += 1

        # Step 12: If process phase is 1, repeat steps 6-11,
        # else if process phase is 3 terminate process.
        if process_phase == 3:
            break


def get_jobs_from_server(client_id: str, jobs_registry: dict, server_url: str):
    '''
    Job Checker method
    '''

    url = f'{server_url}/job_manager/list'

    logger.info(f'Fetching Job list from Server at {url}')

    jobs = get(url, dict())

    for job in jobs:
        if job not in jobs_registry['job_ids']:
            # TODO: logic to fetch job manifest, and if client included, start job thread
            break


def get_job_manifest(client_id: str, job_id: str, server_url: str) -> dict:
    '''
    Download job manifest from server for [job_id]
    '''

    url = f'{server_url}/job_manager/get'

    logger.info(f'Fetching Job Manifest for [{job_id}] from Server at {url}')

    manifest = get(url, {'job_id': job_id})

    for client in manifest['exec_params']['client_info']:
        if client['client_id'] == client_id:
            # TODO: logic to start new job thread
            break
