'''
Application Logic for Client Management
'''
from helpers.kvstore import kv_get, kv_set
from helpers.semaphore import Semaphore
from helpers.logging import logger


class TrainingJobManager:
    '''
    Training Job Manager Class
    '''

    def __init__(self, project_name: str, num_clients=1, dataset_dist_file='def_1c.py',
                 dataset_id='mnist_def', train_ratio=0.8, test_ratio=0.2, val_ratio=0.0,
                 model_file='simple_nn', dataset_preprocess_file='mnist_def',
                 training_loop_file='simple_loop', test_file='simple_test',
                 learning_rate=0.01, rounds=10, batch_size=64, local_epochs=1):
        '''
        constructor
        '''

        self.modification_lock = Semaphore()

        self.project_name = project_name

        self.dataset_params = {
            'num_clients': num_clients,
            'dataset_dist_file': dataset_dist_file,
            'dataset_id': dataset_id,
            'ratio': {
                'train': train_ratio,
                'test': test_ratio,
                'validation': val_ratio
            }
        }

        self.model_params = {
            'model_file': model_file,
            'dataset_preprocess_file': dataset_preprocess_file,
            'training_loop_file': training_loop_file,
            'test_file': test_file
        }

        self.train_params = {
            'learning_rate': learning_rate,
            'rounds': rounds,
            'batch_size': batch_size,
            'local_epochs': local_epochs
        }

        self.job_params = {
            'client_stage': 0,
            'download_jobsheet': False,
            'download_dataset': False,
            'process_phase': 0,
            'extra_params': dict()
        }

        self.exec_params = {
            'client_info': [
                # {
                #     'client_id': 'client-0',
                #     'status': 0
                # }
            ],
            'client_model_params': [],  # state_dict()
            'central_model_param': None,  # state_dict(),
        }

        self._update_state()

    def _read_state(self):
        payload = kv_get(self.project_name)

        if payload is None:
            self.__init__()
        else:
            self.project_name = payload['project_name']
            self.dataset_params = payload['dataset_params']
            self.model_params = payload['model_params']
            self.train_params = payload['train_params']
            self.job_params = payload['job_params']
            self.exec_params = payload['exec_params']

    def _update_state(self):
        data = {
            'project_name': self.project_name,
            'dataset_params': self.dataset_params,
            'model_params': self.model_params,
            'train_params': self.train_params,
            'job_params': self.job_params,
            'exec_params': self.exec_params
        }
        kv_set(self.project_name, data)

    def allow_jobsheet_download(self) -> bool:
        '''
        Allows to download Jobsheet for clients to prepare themselves.
        Basically modifies the job_params.download_jobsheet from False to True.
        Only work if process_phase is 0, client_stage is 0, download_jobsheet is False and download_dataset is False. 
        '''
        # method prefixed with locking and reading state
        self.modification_lock.acquire()
        self._read_state()
        exec_status = True

        # method logic
        if self.job_params['process_phase'] == 0 and self.job_params['client_stage'] == 0 and self.job_params['download_jobsheet'] is False and self.job_params['download_dataset'] is False:
            self.job_params['download_jobsheet'] = True

            # method suffixed with update state and lock release
            self._update_state()
        else:
            # log output and set execution status to False
            logger.warning(
                f'''Cannot ALLOW JobSheet Download! 
                job_params.process_phase is {self.job_params["process_phase"]}, 
                job_params.client_stage is {self.job_params["client_stage"]},
                job_params.download_jobsheet is {self.job_params["download_jobsheet"]}, 
                job_params.download_dataset is {self.job_params["download_dataset"]}.''')
            exec_status = False

        # method suffixed with update state and lock release
        self.modification_lock.release()
        return exec_status

    def allow_dataset_download(self) -> bool:
        '''
        Allows to download dataset for clients to prepare themselves from the data-distributor.
        Basically modifies the job_params.download_dataset from False to True.
        Only work if process_phase is 0, client_stage is 1, download_jobsheet is True and download_dataset is False.
        '''
        # method prefixed with locking and reading state
        self.modification_lock.acquire()
        self._read_state()
        exec_status = True

        # method logic
        if self.job_params['process_phase'] == 0 and self.job_params['client_stage'] == 1 and self.job_params['download_jobsheet'] is True and self.job_params['download_dataset'] is False:
            self.job_params['download_dataset'] = True

            # method suffixed with update state and lock release
            self._update_state()
        else:
            # log output and set execution status to False
            logger.warning(
                f'''Cannot ALLOW Dataset Download! 
                job_params.process_phase is {self.job_params["process_phase"]}, 
                job_params.client_stage is {self.job_params["client_stage"]},
                job_params.download_jobsheet is {self.job_params["download_jobsheet"]}, 
                job_params.download_dataset is {self.job_params["download_dataset"]}.''')
            exec_status = False

        # method suffixed with update state and lock release
        self.modification_lock.release()
        return exec_status

    def add_client(self, client_id: str) -> bool:
        '''
        Adds a Client to the list of clients for the current job, only if job_params.process_phase is 0.
        '''
        # method prefixed with locking and reading state
        self.modification_lock.acquire()
        self._read_state()
        exec_status = True

        # method logic
        if self.job_params['process_phase'] == 0:
            self.exec_params['client_info'].append({
                'client_id': client_id,
                'status': 0
            })

            # method suffixed with update state and lock release
            self._update_state()
        else:
            # log output and set execution status to False
            logger.warning(
                f'''Cannot ADD Client! 
                job_params.process_phase is {self.job_params["process_phase"]}.''')
            exec_status = False

        # method suffixed with update state and lock release
        self.modification_lock.release()
        return exec_status

    def update_client_status(self, client_id: str, client_status: int) -> bool:
        '''
        Updates the status of a client, based on their client_id and 
        if all clients have the same status, the global status, i.e., job_params.client_stage is set as the status of the clients
        '''
        # method prefixed with locking and reading state
        self.modification_lock.acquire()
        self._read_state()
        exec_status = True

        # method logic
        all_client_status = set()

        for i in range(len(self.exec_params['client_info'])):
            # find the client and update their status
            if self.exec_params['client_info'][i]['client_id'] == client_id:
                self.exec_params['client_info'][i]['status'] = client_status

            # collect the status of all the clients
            all_client_status.add(
                self.exec_params['client_info'][i]['status'])

        if len(all_client_status) == 1:
            self.job_params['client_stage'] = list(all_client_status)[0]

        # method suffixed with update state and lock release
        self._update_state()
        self.modification_lock.release()
        return exec_status

    def set_central_model_params(self, params: dict) -> bool:
        '''
        Set or Update the Central Mode Parameters, for initial time, or aggregated update time.
        Only set, if Provess Phase is 0 or 2, i.e., TrainingNotStarted or InCentralAggregation.
        '''
        # method prefixed with locking and reading state
        self.modification_lock.acquire()
        self._read_state()
        exec_status = True

        # method logic
        if self.job_params['process_phase'] == 0 or self.job_params['process_phase'] == 2:
            self.exec_params['central_model_param'] = params

            # method suffixed with update state and lock release
            self._update_state()
        else:
            logger.warning(
                f'Central model parameters NOT SET! job_params.process_phase is {self.job_params["process_phase"]}.')
            exec_status = False

        # method suffixed with update state and lock release
        self.modification_lock.release()
        return exec_status

    def allow_start_training(self) -> bool:
        '''
        Signal clients to start training by setting job_params.process_phase to 1.
        Only works if job_params.client_stage=2 and job_params.process_phase=0
        '''
        # method prefixed with locking and reading state
        self.modification_lock.acquire()
        self._read_state()
        exec_status = True

        # method logic
        if self.job_params['process_phase'] == 0 and self.job_params['client_stage'] == 2:
            self.job_params['process_phase'] = 1

            # method suffixed with update state and lock release
            self._update_state()
        else:
            logger.warning(
                f'Cannot SET process_phase to 1 (InLocalTraining)! job_params.process_phase is {self.job_params["process_phase"]}, job_params.client_stage is {self.job_params["client_stage"]}.')
            exec_status = False

        # method suffixed with update state and lock release
        self.modification_lock.release()
        return exec_status

    def append_client_params(self, client_params: dict) -> bool:
        '''
        Append Trained Model Params from Clients to the exec_params.client_model_params[].
        Only works if job_params.client_stage=3 and job_params.process_phase=1.
        '''
        # method prefixed with locking and reading state
        self.modification_lock.acquire()
        self._read_state()
        exec_status = True

        # method logic
        if self.job_params['process_phase'] == 1 and self.job_params['client_stage'] == 3:
            # add client submitted parameters
            self.exec_params['client_model_params'].append(client_params)

            # if all the client's parameters are submitted, set process_phase to 2, i.e., InCentralAggregation
            if len(self.exec_params['client_model_params']) == self.dataset_params['num_clients']:
                self.job_params['process_phase'] = 2

            # method suffixed with update state and lock release
            self._update_state()
        else:
            logger.warning(
                f'Cannot APPEND client model params! job_params.process_phase is {self.job_params["process_phase"]}, job_params.client_stage is {self.job_params["client_stage"]}.')
            exec_status = False

        # method suffixed with update state and lock release
        self.modification_lock.release()
        return exec_status

    def terminate_training(self) -> None:
        '''
        Signal to Terminate the Training process. This sets the process_phase to 3, i.e., TrainingComplete
        Requires Clients to be waiting for Model Params, i.e., client_stage is 4,
        and process stage is at InCentralAggregation, i.e., process_phase is 2.
        '''
        # method prefixed with locking and reading state
        self.modification_lock.acquire()
        self._read_state()
        exec_status = True

        # method logic
        if self.job_params['process_phase'] == 2 and self.job_params['client_stage'] == 4:
            self.job_params['process_phase'] = 3

            # method suffixed with update state and lock release
            self._update_state()
        else:
            logger.warning(
                f'Cannot Terminate Training Process! job_params.process_phase is {self.job_params["process_phase"]}, job_params.client_stage is {self.job_params["client_stage"]}.')
            exec_status = False

        # method suffixed with update state and lock release
        self.modification_lock.release()
        return exec_status
