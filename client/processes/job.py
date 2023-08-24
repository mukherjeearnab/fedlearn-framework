'''
Job Process Management module
'''
from multiprocessing import Process
from helpers.logging import logger
from helpers.file import create_dir_struct
from helpers.http import get, download_file
from helpers.client_status import update_client_status
from helpers.server_listeners import listen_to_dataset_download_flag, listen_to_start_training
from helpers.server_listeners import download_global_params, upload_client_params
from helpers.server_listeners import listen_for_central_aggregation, listen_for_param_download_training
from helpers.converters import get_state_dict, set_state_dict, tensor_to_data_loader
from helpers.torch import get_device
from processes.training import data_preprocessing, init_model, parameter_mixing
from processes.training import train_model
from processes.tester import test_model


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

    # Step 0: Select Device
    device = get_device()

    # Step 1: ACK of job manifest to server, and update client status to 1.
    update_client_status(client_id, job_id, 1, server_url)

    # Step 2: Listen to download_dataset to turn true, then download dataset.
    # 2.1 listen to download dataset flag
    listen_to_dataset_download_flag(job_id, server_url)

    # 2.2 create the directory structure for the download
    dataset_path = f'./datasets/{job_id}'
    create_dir_struct(dataset_path)

    # 2.3 download dataset to ./datasets/[job_id]/dataset.tuple
    download_file(f'{server_url}/job_manager/download_dataset?job_id={job_id}&client_id={client_id}',
                  f'{dataset_path}/dataset.tuple')

    # Step 3: Preprocess dataset.
    # 3.1 preprocess dataset
    (train_set, test_set) = data_preprocessing(dataset_path,
                                               job_manifest['client_params']['dataset']['preprocessor']['content'],
                                               list(job_manifest['client_params']['train_test_split'].values()))
    # 3.2 create DataLoader Objects for train and test sets
    train_loader = tensor_to_data_loader(train_set,
                                         job_manifest['client_params']['train_params']['batch_size'])
    test_loader = tensor_to_data_loader(test_set,
                                        job_manifest['client_params']['train_params']['batch_size'])

    # Step 4: ACK of dataset to server, and update client status to 2.
    update_client_status(client_id, job_id, 2, server_url)

    # It is a good idea to initialize the model with initial params here.
    model = init_model(job_manifest['client_params']
                       ['model_params']['model_file']['content'])
    # obtain parameters of the model
    previous_params = get_state_dict(model)

    # Step 5: Listen to check when process phase turns 1.
    listen_to_start_training(job_id, server_url)

    # Step 6: Download global parameters from server.
    global_params = download_global_params(job_id, server_url)
    previous_params = global_params

    # some logging vars
    global_round = 1

    # round loop for steps 6-12
    while True:

        # Step 7: ACK of global parameters to server, and update client status to 3.
        update_client_status(client_id, job_id, 3, server_url)

        # Step 8: Perform local training.
        # Step 8.1: Perform the Parameter Mixing
        curr_params = parameter_mixing(global_params, previous_params,
                                       job_manifest['client_params']['model_params']['parameter_mixer']['content'])

        # Step 8.2: Update the model parameters
        set_state_dict(model, curr_params)

        # Step 8.3: Training Loop
        train_model(job_manifest, train_loader, model, device)

        # Step 8.4: Obtain trained model parameters
        curr_params = get_state_dict(model)

        # Step 8.5: Test the trained model parameters with test dataset
        metrics = test_model(model, test_loader, device)
        # as of now, only print the metrics
        print(metrics)

        # Step 9: Send back locally trained model parameters
        # and update client status to 4 on the server automatically.
        upload_client_params(curr_params, client_id, job_id, server_url)
        update_client_status(client_id, job_id, 4, server_url)

        # Step 10: Listen to check when process phase change to 2.
        listen_for_central_aggregation(job_id, server_url)

        # Step 11: Listen to check when process phase change to 1 or 3.
        process_phase = listen_for_param_download_training(job_id, server_url)

        # update round count
        global_round += 1

        # Step 12: If process phase is 1, repeat steps 6-11,
        if process_phase == 1:
            # update previous parameters
            previous_params = curr_params

            # Step 6: Download global parameters from server.
            global_params = download_global_params(job_id, server_url)

        # Step 12: else if process phase is 3 terminate process.
        if process_phase == 3:
            break


def get_jobs_from_server(client_id: str, jobs_registry: dict, server_url: str):
    '''
    Job Checker method
    '''

    url = f'{server_url}/job_manager/list'

    logger.info(f'Fetching Job list from Server at {url}')

    jobs = get(url, dict())

    print('JOBLISTGET', jobs)

    for job_id in jobs:
        print('JOBLISTSET', jobs_registry['job_ids'])
        if job_id not in jobs_registry['job_ids']:
            print('JOBLISTSETIF', jobs_registry['job_ids'])
            jobs_registry['job_ids'].append(job_id)
            get_job_manifest(client_id, job_id, server_url)
            jobs_registry['jobs'][job_id]['process'] = job_proc


def get_job_manifest(client_id: str, job_id: str, server_url: str) -> dict:
    '''
    Download job manifest from server for [job_id]
    '''

    url = f'{server_url}/job_manager/get'

    logger.info(f'Fetching Job Manifest for [{job_id}] from Server at {url}')

    manifest = get(url, {'job_id': job_id})

    for client in manifest['exec_params']['client_info']:
        if client['client_id'] == client_id:
            logger.info(f'Starting Job Process for Job [{job_id}]')

            # start new job thread
            job_proc = Process(target=job_process,
                               args=(client_id, job_id, manifest, server_url), name=f'job_{job_id}')

            # start job process
            job_proc.start()

            # return the job process
            # return job_proc
