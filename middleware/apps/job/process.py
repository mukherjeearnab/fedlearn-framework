'''
Job Process Module
'''
from time import time
from helpers.logging import logger
from helpers.torch import get_device
from apps.job.api import load_job, start_job, allow_start_training, terminate_training
from apps.job.management.dataset.downloader import download_upstream_dataset
from apps.job.management.dataset.prepare import dataset_prepare_for_downstream_clients
from apps.job.management.dataset.allow_dataset_download import allow_downstream_dataset_download
from apps.job.management.set_central_model_params import set_downstream_central_model_params
from apps.job.management.wait_for_client_stage import wait_for_client_stage
from apps.job.management.test_loader import load_test_dataset
from apps.job.management.aggregator import aggregate_downstream_params
from apps.job.management.perflog_upload import upload_perflog_metrics
from apps.middleware.status import update_middleware_status
from apps.server.listeners import listen_to_dataset_download_flag, listen_to_client_stage, listen_to_start_training, listen_for_param_download_training
from apps.server.communication import download_global_params, upload_client_params


def job_process(middleware_id: str, job_id: str, job_manifest: dict, server_url: str):
    '''
    The Job Process method
    1. ACK of job manifest to server, and update middleware (client) status to 1.
        0. Listen to Upstream server for Job Manifest.
        1. Download Upstream Job Manifest.
        2. Prepare Middleware Job Manifest.
        3. Serve Job Manifest (middleware) to Downstream Clients.
        4. Wait for Downstream Clients to send ACK of job manifest.
        5. Send ACK of job manifest to Upstream Server.
    2. Listen to download_dataset to turn true, then download dataset.
        0. Listen to Upstream server for Dataset.
        1. Download Upstream Dataset.
        2. Prepare Downstream Client Datasets (prepare chunks, distribute).
        3. Serve Dataset (middleware) to Downstream Clients (set Download flag of middleware).
        4. Wait for Downstream Clients to send ACK of Dataset.
    3. ACK of dataset to Upstream server, and update middleware (client) status to 2 (2.5).
    4. Listen to check when Upstream Process Phase turns 1.
        0. Listen to Upstream Server for Process Phase to turn 1.
        1. Download Global Params from Upstream Server.
        2. Set Global Params for Downstream Clients to download.
        3. Set Middleware Process Phase to 1, for all Downstream Clients.
        4. Wait for Downstream Clients to send ACK and Downstream Client Stage to be 3.
    5. ACK of global parameters to Upstream server, and update middleware (client) status to 3.
    6. Wait for Downstream Clients to Train.
        1. Wait for Downstream Clients to upload their Parameters to Middleware, as Downstream Client Stage will turn 4.
        2. If Downstream Client Stage is 4, SET Downstream Client (middleware) Process Phase to turn 2.
    7. Perform Aggregation of Downstream Client Parameters.
    8. Send back Aggregated Model Parameters to Upstream Server and ACK of model update, and update middleware (client) status to 4.
    (x) 9. Listen to check when Upstream Server Process Phase change to 2.
    10. Listen to check when Upstream Server Process Phase change to 1 or 3.
    11. If Upstream Server Process Phase is 1, repeat steps 4.2-11, 
        else SET Downstream Client (middleware) Process Phase to 3, and terminate process.
    '''

    # Step 0: Select Device
    device = get_device()

    logger.info(f'In Process Thread for Job [{job_id}]')

    # some logging vars
    global_round = 1

    # Step 1. ACK of job manifest to server, and update middleware (client) status to 1.
    #    0. Listen to Upstream server for Job Manifest. (already done)
    #    1. Download Upstream Job Manifest. (already done)
    #    2. Prepare Middleware Job Manifest.
    exec_status = load_job(job_id)

    if not exec_status:
        logger.error('Job Loading Failed. Exiting...')
        return

    #    3. Serve Job Manifest (middleware) to Downstream Clients.
    #    4. Wait for Downstream Clients to send ACK of job manifest.
    start_job(job_id)

    #    5. Send ACK of job manifest to Upstream Server.
    update_middleware_status(middleware_id, job_id, 1, server_url)

    # Step 2. Listen to download_dataset to turn true, then download dataset.
    #    0. Listen to Upstream server for Dataset.
    listen_to_dataset_download_flag(job_id, server_url)

    #    1. Download Upstream Dataset.
    download_upstream_dataset(job_id, middleware_id, server_url)

    #    2. Prepare Downstream Client Datasets (prepare chunks, distribute).
    dataset_prepare_for_downstream_clients(middleware_id, job_id, job_manifest)

    #    2A. Also Load the Test Dataset Loader
    test_loader = load_test_dataset(job_id, job_manifest)

    #    3. Serve Dataset (middleware) to Downstream Clients (set Download flag of middleware).
    #    4. Wait for Downstream Clients to send ACK of Dataset.
    allow_downstream_dataset_download(job_id)

    # Step 3. ACK of dataset to Upstream server, and update middleware (client) status to 2 (2.5).
    update_middleware_status(middleware_id, job_id, 2, server_url)

    # Step 4. Listen to check when Upstream Process Phase turns 1.
    #    0. Listen to Upstream Server for Process Phase to turn 1.
    listen_to_client_stage(2, job_id, server_url)
    listen_to_start_training(job_id, server_url)

    #    1. Download Global Params from Upstream Server.
    global_params = download_global_params(job_id, server_url)
    # previous_params = global_params

    while True:
        # record start time
        start_time = time()

        #  4.2. Set Global Params for Downstream Clients to download.
        set_downstream_central_model_params(job_id, global_params)

        #    3. Set Middleware Process Phase to 1, for all Downstream Clients.
        allow_start_training(job_id)

        #    4. Wait for Downstream Clients to send ACK and Downstream Client Stage to be 3.
        wait_for_client_stage(job_id, 3)

        # Step 5. ACK of global parameters to Upstream server, and update middleware (client) status to 3.
        update_middleware_status(middleware_id, job_id, 3, server_url)

        # Step 6. Wait for Downstream Clients to Train.
        #    1. Wait for Downstream Clients to upload their Parameters to Middleware, as Downstream Client Stage will turn 4.
        wait_for_client_stage(job_id, 4)

        #    2. If Downstream Client Stage is 4, SET Downstream Client (middleware) Process Phase to turn 2.
        # This is Auto Handled in Job Exec and Params Handler

        # Step 7. Perform Aggregation of Downstream Client Parameters.
        curr_params, metrics = aggregate_downstream_params(
            job_id, test_loader, device)

        # Step 8. Send back Aggregated Model Parameters to Upstream Server and ACK of model update, and update middleware (client) status to 4.
        upload_client_params(curr_params, middleware_id, job_id, server_url)
        update_middleware_status(middleware_id, job_id, 4, server_url)

        # caclulate total time for 1 round
        end_time = time()
        # find the time delta for round and convert the microseconds to milliseconds
        time_delta = (end_time - start_time)*1000
        logger.info(f'Total Round Time Delta: {time_delta} ms')

        # Step 8A. Also Upload the Aggregated Model Test Metrics to PerfLogger.
        upload_perflog_metrics(job_id, metrics, curr_params, time_delta)

        # Step 10. Listen to check when Upstream Server Process Phase change to 1 or 3.
        listen_to_client_stage(4, job_id, server_url)
        process_phase, global_round, _ = listen_for_param_download_training(
            job_id, server_url, global_round)

        # Step 11. If Upstream Server Process Phase is 1, repeat steps 4.2-11,
        #          else SET Downstream Client (middleware) Process Phase to 3, and terminate process.
        if process_phase == 1:
            # update previous parameters
            # previous_params = curr_params

            # Step 6: Download global parameters from server.
            global_params = download_global_params(job_id, server_url)

        # else if Process Phase is 3 SET Downstream Client (middleware) Process Phase to 3, and terminate process.
        if process_phase == 3:
            logger.info(f'Job [{job_id}] terminated. Exiting Process.')
            update_middleware_status(middleware_id, job_id, 5, server_url)

            # also terminate job for downstream clients
            terminate_training(job_id)

            break
