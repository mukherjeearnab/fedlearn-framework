'''
Job Process Module
'''
from time import time
from copy import deepcopy
import traceback
from helpers.logging import logger
from helpers.file import create_dir_struct
from helpers.http import download_file
from helpers.converters import get_base64_state_dict, set_base64_state_dict, tensor_to_data_loader
from helpers.torch import get_device
from helpers.perflog import add_record
from helpers.dynamod import load_module
from apps.client.status import update_client_status
from apps.model.training import data_preprocessing, init_model, parameter_mixing, train_model
from apps.server.listeners import listen_to_dataset_download_flag, listen_to_start_training, listen_to_client_stage
from apps.server.communication import download_global_params, upload_client_params
from apps.server.listeners import listen_for_param_download_training


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
    listen_to_dataset_download_flag(job_id, server_url, client_id)

    # 2.2 create the directory structure for the download
    file_name = f'{client_id}.tuple'
    dataset_path = f'./datasets/{job_id}'
    create_dir_struct(dataset_path)

    # 2.3 download dataset to ./datasets/[job_id]/dataset.tuple
    download_file(f'{server_url}/job_manager/download_dataset?job_id={job_id}&client_id={client_id}',
                  f'{dataset_path}/{file_name}')

    # Step 3: Preprocess dataset.
    # 3.1 preprocess dataset
    try:
        (train_set, test_set) = data_preprocessing(file_name, dataset_path,
                                                   job_manifest['client_params']['dataset']['preprocessor']['content'])
        #    list(job_manifest['client_params']['train_test_split'].values()))
    except Exception:
        logger.error(
            f'Failed to run Dataset Proprocessing. Aborting Process for job [{job_id}]!\n{traceback.format_exc()}')
        update_client_status(client_id, job_id, 5, server_url)
        exit()

    # 3.2 create DataLoader Objects for train and test sets
    train_loader = tensor_to_data_loader(train_set,
                                         job_manifest['client_params']['train_params']['batch_size'])
    test_loader = tensor_to_data_loader(test_set,
                                        job_manifest['client_params']['train_params']['batch_size'])

    # Step 4: ACK of dataset to server, and update client status to 2.
    update_client_status(client_id, job_id, 2, server_url)

    # wait for client stage be 2
    listen_to_client_stage(2, job_id, server_url, client_id)

    # It is a good idea to initialize the local and global model with initial params here.
    try:
        local_model = init_model(job_manifest['client_params']
                                 ['model_params']['model_file']['content'])
    except Exception:
        logger.error(
            f'Failed to init Model. Aborting Process for job [{job_id}]!\n{traceback.format_exc()}')
        update_client_status(client_id, job_id, 5, server_url)
        exit()

    global_model = deepcopy(local_model)
    prev_local_model = deepcopy(local_model)

    # obtain parameters of the model
    previous_params = get_base64_state_dict(prev_local_model)

    # Step 5: Listen to check when process phase turns 1.
    listen_to_start_training(job_id, server_url, client_id)

    # Step 6: Download global parameters from server.
    global_params = download_global_params(job_id, server_url)
    # previous_params = global_params

    # some logging vars
    global_round = 1
    total_rounds = job_manifest['server_params']['train_params']['rounds']

    extra_data = {'round_info': {'total_rounds': total_rounds}}

    # record start time
    start_time = time()

    # round loop for steps 6-12
    while True:

        # Step 7: ACK of global parameters to server, and update client status to 3.
        update_client_status(client_id, job_id, 3, server_url)

        # wait for client stage be 3
        listen_to_client_stage(3, job_id, server_url, client_id)

        # Step 8: Perform local training.
        # Step 8.1: Perform the Parameter Mixing
        try:
            curr_params = parameter_mixing(global_params, previous_params,
                                           job_manifest['client_params']['model_params']['parameter_mixer']['content'])
        except Exception:
            logger.error(
                f'Failed to run Parameter Mixer. Aborting Process for job [{job_id}]!\n{traceback.format_exc()}')
            update_client_status(client_id, job_id, 5, server_url)
            break

        # Step 8.2.1: Update the local model parameters
        set_base64_state_dict(local_model, curr_params)

        # Step 8.2.2: Update the local model parameters
        set_base64_state_dict(global_model, global_params)

        # Step 8.2.3: Update the local model parameters
        set_base64_state_dict(prev_local_model, previous_params)

        # Step 8.3: Training Loop
        extra_data['round_info']['current_round'] = global_round
        try:
            train_model(job_manifest, train_loader,
                        local_model, global_model, prev_local_model, extra_data, device)
        except Exception:
            logger.error(
                f'Failed to train Model. Aborting Process for job [{job_id}]!\n{traceback.format_exc()}')
            update_client_status(client_id, job_id, 5, server_url)
            break

        # Step 8.4: Obtain trained model parameters
        curr_params = get_base64_state_dict(local_model)

        # Step 8.5: Test the trained model parameters with test dataset
        try:
            testing_module = load_module(
                'testing_module', job_manifest['client_params']['model_params']['test_file']['content'])
            metrics = testing_module.test_runner(
                local_model, test_loader, device)
        except Exception:
            logger.error(
                f'Failed to run Testing Script. Aborting Process for job [{job_id}]!\n{traceback.format_exc()}')
            update_client_status(client_id, job_id, 5, server_url)
            break

        # caclulate total time for 1 round
        end_time = time()
        # find the time delta for round and convert the microseconds to milliseconds
        time_delta = (end_time - start_time)*1000
        logger.info(f'Total Round Time Delta: {time_delta} ms')
        # report metrics to PerfLog Server
        add_record(client_id, job_id, metrics, global_round, time_delta)

        # Step 9: Send back locally trained model parameters
        # and update client status to 4 on the server automatically.
        upload_client_params(curr_params, client_id, job_id, server_url)
        update_client_status(client_id, job_id, 4, server_url)

        # wait for client stage be 4
        listen_to_client_stage(4, job_id, server_url, client_id)

        # Step 10: Listen to check when process phase change to 2.
        # listen_for_central_aggregation(job_id, server_url)

        # Step 11: Listen to check when process phase change to 1 or 3.
        process_phase, global_round, abort_signal = listen_for_param_download_training(
            job_id, server_url, global_round, client_id)

        # update round count
        # global_round += 1

        # if abort signal is true, abort the job
        if abort_signal:
            logger.info(f'Job [{job_id}] Aborted. Exiting Process.')
            update_client_status(client_id, job_id, 5, server_url)
            break

        # Step 12: If process phase is 1, repeat steps 6-11,
        if process_phase == 1:
            # update previous parameters
            previous_params = curr_params

            # Step 6: Download global parameters from server.
            global_params = download_global_params(job_id, server_url)

        # Step 12: else if process phase is 3 terminate process.
        if process_phase == 3:
            logger.info(f'Job [{job_id}] terminated. Exiting Process.')
            update_client_status(client_id, job_id, 5, server_url)
            break

        # record start time
        start_time = time()
